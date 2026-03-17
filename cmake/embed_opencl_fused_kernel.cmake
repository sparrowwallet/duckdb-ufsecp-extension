# embed_opencl_fused_kernel.cmake -- Generate opencl_fused_kernel_source.h
#
# Inputs (passed via -D):
#   KERNEL_DIR  -- Path to UltrafastSecp256k1/opencl/kernels/
#   OUTPUT_FILE -- Path to write opencl_fused_kernel_source.h
#
# Reads the 4 required kernel files (field, point, extended, affine),
# strips #include and #pragma once lines, and appends the fused
# bip352_fused_kernel definition. Writes the result as a C++ raw string literal.

set(KERNEL_FILES
    "${KERNEL_DIR}/secp256k1_field.cl"
    "${KERNEL_DIR}/secp256k1_point.cl"
    "${KERNEL_DIR}/secp256k1_extended.cl"
    "${KERNEL_DIR}/secp256k1_affine.cl"
)

set(COMBINED "")

# Read and strip each kernel file
foreach(FILE ${KERNEL_FILES})
    file(READ "${FILE}" CONTENT)
    # Strip #include lines
    string(REGEX REPLACE "#include [^\n]*\n" "" CONTENT "${CONTENT}")
    # Strip #pragma once lines
    string(REGEX REPLACE "#pragma once[^\n]*\n" "" CONTENT "${CONTENT}")
    # Strip include guards (#ifndef/#define/#endif for header guards)
    string(REGEX REPLACE "#ifndef SECP256K1_[A-Z_]*_CL\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#define SECP256K1_[A-Z_]*_CL\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#endif // SECP256K1_[A-Z_]*_CL\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#endif [^\n]*SECP256K1_[A-Z_]*_CL[^\n]*\n" "" CONTENT "${CONTENT}")
    string(APPEND COMBINED "${CONTENT}\n")
endforeach()

# Append fused BIP-352 kernel definition
string(APPEND COMBINED "
// =============================================================================
// Fused BIP-352 kernel -- entire per-row pipeline in one thread
// =============================================================================

__kernel void bip352_fused_kernel(
    __global const uchar *tweak_xy,     // N x 64 bytes (LE)
    __constant uchar *scan_key,         // 32 bytes (LE)
    __global uchar *out_x,              // N x 32 bytes (LE output)
    __global uchar *out_y,              // N x 32 bytes (LE output)
    __constant uint *tag_midstate,      // 8 x uint32_t
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // ------------------------------------------------------------------
    // Phase 0: Load inputs (LE wire format -> internal representation)
    // ------------------------------------------------------------------

    // Load tweak point: LE bytes -> FieldElement (4x ulong LE limbs)
    __global const uchar *tweak = tweak_xy + gid * 64;
    FieldElement fx, fy;
    for (int i = 0; i < 4; i++) {
        ulong lx = 0, ly = 0;
        for (int j = 7; j >= 0; j--) {
            lx = (lx << 8) | tweak[i * 8 + j];
            ly = (ly << 8) | tweak[32 + i * 8 + j];
        }
        fx.limbs[i] = lx;
        fy.limbs[i] = ly;
    }

    AffinePoint tweak_pt;
    tweak_pt.x = fx;
    tweak_pt.y = fy;

    // Load scan key: 32 LE bytes -> reverse to 32 BE bytes -> scalar_from_bytes
    uchar sk_be[32];
    for (int i = 0; i < 32; i++) sk_be[i] = scan_key[31 - i];
    Scalar sk;
    scalar_from_bytes_impl(sk_be, &sk);

    // ------------------------------------------------------------------
    // Phase 1: shared_secret = scan_key * tweak_point
    // ------------------------------------------------------------------
    JacobianPoint shared_jac;
    scalar_mul_glv_impl(&shared_jac, &sk, &tweak_pt);

    // ------------------------------------------------------------------
    // Phase 2: Jacobian -> affine -> SEC1 compressed serialization
    // ------------------------------------------------------------------
    AffinePoint shared_aff;
    jacobian_to_affine_convert_impl(&shared_aff,
        &shared_jac.x, &shared_jac.y, &shared_jac.z);

    uchar x_bytes[32], y_bytes[32];
    field_to_bytes_impl(&shared_aff.x, x_bytes);
    field_to_bytes_impl(&shared_aff.y, y_bytes);

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
    ctx.total_len = 64;  // 64 bytes already processed by midstate
    sha256_update(&ctx, ser, 37);
    uchar hash[32];
    sha256_final(&ctx, hash);

    // ------------------------------------------------------------------
    // Phase 4: output_point = hash * G
    // ------------------------------------------------------------------
    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);
    JacobianPoint out_jac;
    scalar_mul_generator_windowed_impl(&out_jac, &hs);

    AffinePoint out_aff;
    jacobian_to_affine_convert_impl(&out_aff,
        &out_jac.x, &out_jac.y, &out_jac.z);

    // ------------------------------------------------------------------
    // Phase 5: Write output as LE bytes
    // ------------------------------------------------------------------
    uchar ox_be[32], oy_be[32];
    field_to_bytes_impl(&out_aff.x, ox_be);
    field_to_bytes_impl(&out_aff.y, oy_be);

    __global uchar *dst_x = out_x + gid * 32;
    __global uchar *dst_y = out_y + gid * 32;
    for (int i = 0; i < 32; i++) {
        dst_x[i] = ox_be[31 - i];
        dst_y[i] = oy_be[31 - i];
    }
}
")

# Write as C++ raw string literal header
file(WRITE "${OUTPUT_FILE}" "// Auto-generated by cmake/embed_opencl_fused_kernel.cmake -- do not edit\n")
file(APPEND "${OUTPUT_FILE}" "#pragma once\n\n")
file(APPEND "${OUTPUT_FILE}" "static const char OPENCL_FUSED_KERNEL_SOURCE[] = R\"opencl_src(\n")
file(APPEND "${OUTPUT_FILE}" "${COMBINED}")
file(APPEND "${OUTPUT_FILE}" ")opencl_src\";\n")
