# embed_metal_shaders.cmake -- Generate metal_shader_source.h with embedded shader source
#
# Inputs (passed via -D):
#   SHADER_DIR  -- Path to UltrafastSecp256k1/metal/shaders/
#   OUTPUT_FILE -- Path to write metal_shader_source.h
#
# Reads the 5 required shader headers (field, point, bloom, extended, hash160),
# strips #include and #pragma once lines, prepends Metal stdlib header, and
# appends inline scalar_mul_batch + generator_mul_batch kernel definitions.
# Writes the result as a C++ raw string literal.

set(SHADER_FILES
    "${SHADER_DIR}/secp256k1_field.h"
    "${SHADER_DIR}/secp256k1_point.h"
    "${SHADER_DIR}/secp256k1_bloom.h"
    "${SHADER_DIR}/secp256k1_extended.h"
    "${SHADER_DIR}/secp256k1_hash160.h"
)

set(COMBINED "")

# Prepend Metal stdlib
string(APPEND COMBINED "#include <metal_stdlib>\nusing namespace metal;\n\n")

# Read and strip each shader file
foreach(FILE ${SHADER_FILES})
    file(READ "${FILE}" CONTENT)
    # Strip #include lines
    string(REGEX REPLACE "#include [^\n]*\n" "" CONTENT "${CONTENT}")
    # Strip #pragma once lines
    string(REGEX REPLACE "#pragma once[^\n]*\n" "" CONTENT "${CONTENT}")
    string(APPEND COMBINED "${CONTENT}\n")
endforeach()

# Append inline kernel definitions (matching secp256k1_kernels.metal signatures)
string(APPEND COMBINED "
// =============================================================================
// Inline kernel definitions for BIP-352 scanning
// =============================================================================

kernel void scalar_mul_batch(
    device const AffinePoint *bases    [[buffer(0)]],
    device const Scalar256 *scalars    [[buffer(1)]],
    device AffinePoint *results        [[buffer(2)]],
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    AffinePoint base = bases[tid];
    Scalar256 k = scalars[tid];
    JacobianPoint jac = scalar_mul_glv(base, k);
    results[tid] = jacobian_to_affine(jac);
}

kernel void generator_mul_batch(
    device const Scalar256 *scalars    [[buffer(0)]],
    device AffinePoint *results        [[buffer(1)]],
    constant uint &count               [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    AffinePoint gen = generator_affine();
    Scalar256 k = scalars[tid];
    JacobianPoint jac = scalar_mul(gen, k);
    results[tid] = jacobian_to_affine(jac);
}

// =============================================================================
// Fused BIP-352 kernel -- entire per-row pipeline in one thread
// =============================================================================

kernel void bip352_fused_kernel(
    device const uchar *tweak_xy     [[buffer(0)]],   // N x 64 bytes (LE)
    constant uchar *scan_key         [[buffer(1)]],   // 32 bytes (LE)
    device uchar *out_x              [[buffer(2)]],   // N x 32 bytes (LE output)
    device uchar *out_y              [[buffer(3)]],   // N x 32 bytes (LE output)
    constant uint *tag_midstate      [[buffer(4)]],   // 8 x uint32_t
    constant uint &count             [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    // ------------------------------------------------------------------
    // Phase 0: Load inputs (LE wire format -> internal representation)
    // ------------------------------------------------------------------

    // Load tweak point: LE bytes -> FieldElement (8x uint32 LE limbs)
    // Each u32 limb = 4 consecutive LE bytes
    device const uchar *tweak = tweak_xy + tid * 64;
    FieldElement fx, fy;
    for (int i = 0; i < 8; i++) {
        int base = i * 4;
        fx.limbs[i] = (uint(tweak[base]) | (uint(tweak[base+1]) << 8) |
                       (uint(tweak[base+2]) << 16) | (uint(tweak[base+3]) << 24));
        fy.limbs[i] = (uint(tweak[32 + base]) | (uint(tweak[32 + base+1]) << 8) |
                       (uint(tweak[32 + base+2]) << 16) | (uint(tweak[32 + base+3]) << 24));
    }
    AffinePoint tweak_pt;
    tweak_pt.x = fx;
    tweak_pt.y = fy;

    // Load scan key: 32 LE bytes -> reverse to 32 BE bytes -> scalar_from_bytes
    uchar sk_be[32];
    for (int i = 0; i < 32; i++) sk_be[i] = scan_key[31 - i];
    Scalar256 sk = scalar_from_bytes(sk_be);

    // ------------------------------------------------------------------
    // Phase 1: shared_secret = scan_key * tweak_point
    // ------------------------------------------------------------------
    JacobianPoint shared_jac = scalar_mul_glv(tweak_pt, sk);
    AffinePoint shared_aff = jacobian_to_affine(shared_jac);

    // ------------------------------------------------------------------
    // Phase 2: Jacobian -> affine -> SEC1 compressed serialization
    // ------------------------------------------------------------------
    uchar x_bytes[32], y_bytes[32];
    field_to_bytes(shared_aff.x, x_bytes);
    field_to_bytes(shared_aff.y, y_bytes);

    uchar ser[37];
    ser[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 32; i++) ser[i + 1] = x_bytes[i];
    ser[33] = 0; ser[34] = 0; ser[35] = 0; ser[36] = 0;

    // ------------------------------------------------------------------
    // Phase 3: Tagged SHA-256 with BIP0352/SharedSecret midstate
    // ------------------------------------------------------------------
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = tag_midstate[i];
    ctx.buf_len = 0;
    ctx.total_len_lo = 64;
    ctx.total_len_hi = 0;
    sha256_update(ctx, ser, 37);
    uchar hash[32];
    sha256_final(ctx, hash);

    // ------------------------------------------------------------------
    // Phase 4: output_point = hash * G
    // ------------------------------------------------------------------
    Scalar256 hs = scalar_from_bytes(hash);
    AffinePoint gen = generator_affine();
    JacobianPoint out_jac = scalar_mul_glv(gen, hs);
    AffinePoint out_aff = jacobian_to_affine(out_jac);

    // ------------------------------------------------------------------
    // Phase 5: Write output as LE bytes
    // ------------------------------------------------------------------
    // field_to_bytes gives big-endian; reverse to LE wire format
    uchar ox_be[32], oy_be[32];
    field_to_bytes(out_aff.x, ox_be);
    field_to_bytes(out_aff.y, oy_be);

    device uchar *dst_x = out_x + tid * 32;
    device uchar *dst_y = out_y + tid * 32;
    for (int i = 0; i < 32; i++) {
        dst_x[i] = ox_be[31 - i];
        dst_y[i] = oy_be[31 - i];
    }
}
")

# Write as C++ raw string literal header
file(WRITE "${OUTPUT_FILE}" "// Auto-generated by cmake/embed_metal_shaders.cmake -- do not edit\n")
file(APPEND "${OUTPUT_FILE}" "#pragma once\n\n")
file(APPEND "${OUTPUT_FILE}" "static const char METAL_SHADER_SOURCE[] = R\"metal_src(\n")
file(APPEND "${OUTPUT_FILE}" "${COMBINED}")
file(APPEND "${OUTPUT_FILE}" ")metal_src\";\n")
