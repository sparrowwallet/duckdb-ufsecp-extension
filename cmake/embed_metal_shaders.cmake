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

// scalar_mul_generator_lut is defined in secp256k1_extended.h (included above)

constant int GEN_LUT_N = 65536;
constant int GEN_LUT_SLICES = 16;

// =============================================================================
// Generator LUT build kernels
// =============================================================================

// Kernel 1: Compute base points B_i = 2^(16*i) * G for i=0..15
kernel void compute_lut_base_points(
    device AffinePoint *bases    [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;

    bases[0] = generator_affine();

    JacobianPoint p;
    AffinePoint g = generator_affine();
    p.x = g.x; p.y = g.y; p.z = field_one(); p.infinity = 0;

    for (int i = 1; i < GEN_LUT_SLICES; i++) {
        for (int d = 0; d < 16; d++)
            p = jacobian_double(p);

        AffinePoint aff = jacobian_to_affine(p);
        bases[i] = aff;

        p.x = aff.x; p.y = aff.y; p.z = field_one(); p.infinity = 0;
    }
}

// Kernel 2: Build LUT entries via sequential chain + serial inversion
// One threadgroup per slice, 1 thread each.
kernel void gen_lut_build_affine(
    device const AffinePoint *bases      [[buffer(0)]],
    device AffinePoint *aff_table        [[buffer(1)]],
    device FieldElement *h_buf           [[buffer(2)]],
    constant int &n_entries              [[buffer(3)]],
    uint slice [[threadgroup_position_in_grid]])
{
    if (slice >= (uint)GEN_LUT_SLICES) return;

    int offset = int(slice) * n_entries;
    device FieldElement *h = h_buf + int(slice) * n_entries;

    aff_table[offset].x = field_zero();
    aff_table[offset].y = field_zero();

    aff_table[offset + 1] = bases[slice];

    AffinePoint base = bases[slice];
    JacobianPoint acc;
    acc.x = base.x; acc.y = base.y; acc.z = field_one(); acc.infinity = 0;

    for (int j = 2; j < n_entries; j++) {
        FieldElement h_val;
        acc = jacobian_add_mixed_h(acc, base, h_val);
        h[j - 2] = h_val;
        aff_table[offset + j].x = acc.x;
        aff_table[offset + j].y = acc.y;
    }

    FieldElement z_inv = field_inv(acc.z);

    for (int j = n_entries - 1; j >= 2; --j) {
        FieldElement h_save;
        if (j > 2) h_save = h[j - 2];
        h[j - 2] = z_inv;
        if (j > 2) z_inv = field_mul(h_save, z_inv);
    }
}

// Kernel 3: Parallel affine conversion using precomputed Z^{-1}
kernel void gen_lut_convert_zinv(
    device AffinePoint *aff_table        [[buffer(0)]],
    device const FieldElement *h_buf     [[buffer(1)]],
    constant int &n_entries              [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    int per_slice = n_entries - 2;
    int total = GEN_LUT_SLICES * per_slice;
    if (gid >= (uint)total) return;

    int slice = int(gid) / per_slice;
    int j = (int(gid) % per_slice) + 2;
    int offset = slice * n_entries;
    device const FieldElement *h = h_buf + slice * n_entries;

    FieldElement zi = h[j - 2];
    FieldElement z_inv2 = field_sqr(zi);
    FieldElement z_inv3 = field_mul(zi, z_inv2);

    FieldElement px = aff_table[offset + j].x;
    FieldElement py = aff_table[offset + j].y;
    aff_table[offset + j].x = field_mul(px, z_inv2);
    aff_table[offset + j].y = field_mul(py, z_inv3);
}

// =============================================================================
// Precomputed generator nibble tables for GLV generator multiplication
// =============================================================================
// GENERATOR_TABLE_NIBBLE[i] = i*G  (i=0..15)
// GENERATOR_TABLE_NIBBLE_PHI[i] = phi(i*G) where phi is the GLV endomorphism
// Values stored as 8x uint32 LE limbs per FieldElement (Metal native format)

constant AffinePoint GENERATOR_TABLE_NIBBLE[16] = {
    {{{0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u}},{{0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u}}},
    {{{0x16f81798u,0x59f2815bu,0x2dce28d9u,0x029bfcdbu,0xce870b07u,0x55a06295u,0xf9dcbbacu,0x79be667eu}},{{0xfb10d4b8u,0x9c47d08fu,0xa6855419u,0xfd17b448u,0x0e1108a8u,0x5da4fbfcu,0x26a3c465u,0x483ada77u}}},
    {{{0x5c709ee5u,0xabac09b9u,0x8cef3ca7u,0x5c778e4bu,0x95c07cd8u,0x3045406eu,0x41ed7d6du,0xc6047f94u}},{{0x50cfe52au,0x236431a9u,0x3266d0e1u,0xf7f63265u,0x466ceaeeu,0xa3c58419u,0xa63dc339u,0x1ae168feu}}},
    {{{0xbce036f9u,0x8601f113u,0x836f99b0u,0xb531c845u,0xf89d5229u,0x49344f85u,0x9258c310u,0xf9308a01u}},{{0x84b8e672u,0x6cb9fd75u,0x34c2231bu,0x6500a999u,0x2a37f356u,0x0fe337e6u,0x632de814u,0x388f7b0fu}}},
    {{{0xe8c4cd13u,0x74fa94abu,0x0ee07584u,0xcc6c1390u,0x930b1404u,0x581e4904u,0xc10d80f3u,0xe493dbf1u}},{{0x47739922u,0xcfe97bdcu,0xbfbdfe40u,0xd967ae33u,0x8ea51448u,0x5642e209u,0xa0d455b7u,0x51ed993eu}}},
    {{{0xb240efe4u,0xcba8d569u,0xdc619ab7u,0xe88b84bdu,0x0a5c5128u,0x55b4a725u,0x1a072093u,0x2f8bde4du}},{{0xa6ac62d6u,0xdca87d3au,0xab0d6840u,0xf788271bu,0xa6c9c426u,0xd4dba9ddu,0x36e5e3d6u,0xd8ac2226u}}},
    {{{0x60297556u,0x2f057a14u,0x8568a18bu,0x82f6472fu,0x355235d3u,0x20453a14u,0x755eeea4u,0xfff97bd5u}},{{0xb075f297u,0x3c870c36u,0x518fe4a0u,0xde80f0f6u,0x7f45c560u,0xf3be9601u,0xacfbb620u,0xae12777au}}},
    {{{0xcac4f9bcu,0xe92bddedu,0x0330e39cu,0x3d419b7eu,0xf2ea7a0eu,0xa398f365u,0x6e5db4eau,0x5cbdf064u}},{{0x087264dau,0xa5082628u,0x13fde7b5u,0xa813d0b8u,0x861a54dbu,0xa3178d6du,0xba255960u,0x6aebca40u}}},
    {{{0xe10a2a01u,0x67784ef3u,0xe5af888au,0x0a1bdd05u,0xb70f3c2fu,0xaff3843fu,0x5cca351du,0x2f01e5e1u}},{{0x6cbde904u,0xb5da2cb7u,0xba5b7617u,0xc2e213d6u,0x132d13b4u,0x293d082au,0x41539949u,0x5c4da8a7u}}},
    {{{0xfc27ccbeu,0xc35f110du,0x4c57e714u,0xe0979697u,0x9f559abdu,0x09ad178au,0xf0c7f653u,0xacd484e2u}},{{0xc64f9c37u,0x05cc262au,0x375f8e0fu,0xadd888a4u,0x763b61e9u,0x64380971u,0xb0a7d9fdu,0xcc338921u}}},
    {{{0x47e247c7u,0x52a68e2au,0x1943c2b7u,0x3442d49bu,0x1ae6ae5du,0x35477c7bu,0x47f3c862u,0xa0434d9eu}},{{0x037368d7u,0x3cbee53bu,0xd877a159u,0x6f794c2eu,0x93a24c69u,0xa3b6c7e6u,0x5419bc27u,0x893aba42u}}},
    {{{0x5da008cbu,0xbbec1789u,0xe5c17891u,0x5649980bu,0x70c65aacu,0x5ef4246bu,0x58a9411eu,0x774ae7f8u}},{{0xc953c61bu,0x301d74c9u,0xdff9d6a8u,0x372db1e2u,0xd7b7b365u,0x0243dd56u,0xeb6b5e19u,0xd984a032u}}},
    {{{0x70afe85au,0xc5b0f470u,0x9620095bu,0x687cf441u,0x4d734633u,0x15c38f00u,0x48e7561bu,0xd01115d5u}},{{0xf4062327u,0x6b051b13u,0xd9a86d52u,0x79238c5du,0xe17bd815u,0xa8b64537u,0xc815e0d7u,0xa9f34ffdu}}},
    {{{0x19405aa8u,0xdeeddf8fu,0x610e58cdu,0xb075fbc6u,0xc3748651u,0xc7d1d205u,0xd975288bu,0xf28773c2u}},{{0xdb03ed81u,0x29b5cb52u,0x521fa91fu,0x3a1a06dau,0x65cdaf47u,0x758212ebu,0x8d880a89u,0x0ab0902eu}}},
    {{{0x60e823e4u,0xe49b241au,0x678949e6u,0x26aa7b63u,0x07d38e32u,0xfd64e67fu,0x895e719cu,0x499fdf9eu}},{{0x03a13f5bu,0xc65f40d4u,0x7a3f95bcu,0x464279c2u,0xa7b3d464u,0x90f044e4u,0xb54e8551u,0xcac2f6c4u}}},
    {{{0xe27e080eu,0x44adbcf8u,0x3c85f79eu,0x31e5946fu,0x095ff411u,0x5a465ae3u,0x7d43ea96u,0xd7924d4fu}},{{0xf6a26b58u,0xc504dc9fu,0xd896d3a5u,0xea40af2bu,0x28cc6defu,0x83842ec2u,0xa86c72a6u,0x581e2872u}}}
};

constant AffinePoint GENERATOR_TABLE_NIBBLE_PHI[16] = {
    {{{0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u}},{{0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u,0x00000000u}}},
    {{{0x00b88fcbu,0xa7bba044u,0x7f15e98du,0x87284406u,0x96902325u,0xab0102b6u,0x9da01887u,0xbcace2e9u}},{{0xfb10d4b8u,0x9c47d08fu,0xa6855419u,0xfd17b448u,0x0e1108a8u,0x5da4fbfcu,0x26a3c465u,0x483ada77u}}},
    {{{0xd89250e1u,0x3e995b6eu,0xe43837efu,0xd2fad8ccu,0x59f87b33u,0x4135ee7du,0xb34ce6dfu,0xc360a6d0u}},{{0x50cfe52au,0x236431a9u,0x3266d0e1u,0xf7f63265u,0x466ceaeeu,0xa3c58419u,0xa63dc339u,0x1ae168feu}}},
    {{{0x77206b2fu,0xf7f0728cu,0xc6dc8e1cu,0x8af1e022u,0x2a28fa2fu,0x8dcd8dcfu,0x731f9b4bu,0xdf6edf03u}},{{0x84b8e672u,0x6cb9fd75u,0x34c2231bu,0x6500a999u,0x2a37f356u,0x0fe337e6u,0x632de814u,0x388f7b0fu}}},
    {{{0x3b306100u,0x5bde5b33u,0xab487127u,0x714c30b5u,0xb90e324bu,0x5c45faf8u,0x0d382907u,0x1b77921fu}},{{0x47739922u,0xcfe97bdcu,0xbfbdfe40u,0xd967ae33u,0x8ea51448u,0x5642e209u,0xa0d455b7u,0x51ed993eu}}},
    {{{0x95a83668u,0x138c6946u,0xe0d097ccu,0xa045693eu,0xccb94671u,0xf79f54fbu,0xacda49dfu,0x337b52e3u}},{{0xa6ac62d6u,0xdca87d3au,0xab0d6840u,0xf788271bu,0xa6c9c426u,0xd4dba9ddu,0x36e5e3d6u,0xd8ac2226u}}},
    {{{0x78f38045u,0x47aaf280u,0x56a15a68u,0x86649d3eu,0xe3e8bed7u,0x5e3aa731u,0xaa535fc6u,0xe63bcdd9u}},{{0xb075f297u,0x3c870c36u,0x518fe4a0u,0xde80f0f6u,0x7f45c560u,0xf3be9601u,0xacfbb620u,0xae12777au}}},
    {{{0x4e53bc94u,0x3bc4686eu,0x0faf7aaau,0x0d3b20e2u,0xc095c06eu,0xa4fec4d1u,0x4bea0b77u,0x13f26e75u}},{{0x087264dau,0xa5082628u,0x13fde7b5u,0xa813d0b8u,0x861a54dbu,0xa3178d6du,0xba255960u,0x6aebca40u}}},
    {{{0x2446cc73u,0x03e94774u,0x24257657u,0xb4ff7715u,0x29e24892u,0xaa77840fu,0x42d401a7u,0x47ab6503u}},{{0x6cbde904u,0xb5da2cb7u,0xba5b7617u,0xc2e213d6u,0x132d13b4u,0x293d082au,0x41539949u,0x5c4da8a7u}}},
    {{{0x65953a52u,0x20cd912eu,0xef6d44e1u,0xb565cdf5u,0xec58ab20u,0x7b6558afu,0x7e44e819u,0x87b40403u}},{{0xc64f9c37u,0x05cc262au,0x375f8e0fu,0xadd888a4u,0x763b61e9u,0x64380971u,0xb0a7d9fdu,0xcc338921u}}},
    {{{0x741afe29u,0xbdb3e957u,0x083762e4u,0xc1938d8eu,0x46813990u,0xa136ebb2u,0xf7a397b1u,0x26ce269bu}},{{0x037368d7u,0x3cbee53bu,0xd877a159u,0x6f794c2eu,0x93a24c69u,0xa3b6c7e6u,0x5419bc27u,0x893aba42u}}},
    {{{0xbb209ce7u,0xc5ff4334u,0x0b5ff620u,0x79859bb7u,0xbebf1a26u,0x8d897c41u,0x171dac1du,0x51f4d3d1u}},{{0xc953c61bu,0x301d74c9u,0xdff9d6a8u,0x372db1e2u,0xd7b7b365u,0x0243dd56u,0xeb6b5e19u,0xd984a032u}}},
    {{{0x042295e5u,0x4a3eb52cu,0xc9535355u,0xf9482837u,0x2eac82adu,0xac154842u,0x953aac41u,0x88591bfdu}},{{0xf4062327u,0x6b051b13u,0xd9a86d52u,0x79238c5du,0xe17bd815u,0xa8b64537u,0xc815e0d7u,0xa9f34ffdu}}},
    {{{0x475fb678u,0x60aaee6au,0x4a3d0562u,0x32907ed7u,0x78fc783bu,0x07046c45u,0x4bb890a2u,0xf14d5837u}},{{0xdb03ed81u,0x29b5cb52u,0x521fa91fu,0x3a1a06dau,0x65cdaf47u,0x758212ebu,0x8d880a89u,0x0ab0902eu}}},
    {{{0x20a0b458u,0x0e6ab7eeu,0x27c529f6u,0x580656a6u,0x87c37384u,0x1548f0dcu,0x7810048au,0x7b125217u}},{{0x03a13f5bu,0xc65f40d4u,0x7a3f95bcu,0x464279c2u,0xa7b3d464u,0x90f044e4u,0xb54e8551u,0xcac2f6c4u}}},
    {{{0x71b1b3b4u,0x3ac0a40cu,0xc1c0a639u,0x05cc3bc9u,0x512b6948u,0x0e1b4825u,0xf5f9454au,0x805f1105u}},{{0xf6a26b58u,0xc504dc9fu,0xd896d3a5u,0xea40af2bu,0x28cc6defu,0x83842ec2u,0xa86c72a6u,0x581e2872u}}}
};

// =============================================================================
// GLV generator multiplication using precomputed constant nibble tables
// =============================================================================

inline int get_nibble_4bit(thread const Scalar256 &s, int pos) {
    int bp = pos * 4;
    int li = bp / 32;
    int sh = bp & 31;
    uint v = s.limbs[li] >> sh;
    if (sh > 28 && li < 7) v |= s.limbs[li + 1] << (32 - sh);
    return int(v & 0xFu);
}

inline JacobianPoint scalar_mul_generator_glv_const(thread const Scalar256 &k) {
    if (scalar256_is_zero(k)) return point_at_infinity();

    Scalar256 k1, k2;
    int k1_neg, k2_neg;
    glv_decompose(k, k1, k2, k1_neg, k2_neg);

    int bl1 = scalar256_bitlen(k1);
    int bl2 = scalar256_bitlen(k2);
    int max_bits = (bl1 > bl2) ? bl1 : bl2;
    int num_windows = (max_bits + 3) / 4;

    JacobianPoint R = point_at_infinity();
    for (int w = num_windows - 1; w >= 0; --w) {
        if (R.infinity == 0) {
            R = jacobian_double(R); R = jacobian_double(R);
            R = jacobian_double(R); R = jacobian_double(R);
        }
        int w1 = get_nibble_4bit(k1, w);
        if (w1) {
            AffinePoint pt = GENERATOR_TABLE_NIBBLE[w1];
            if (k1_neg) pt.y = field_negate(pt.y);
            if (R.infinity != 0) {
                R.x = pt.x; R.y = pt.y; R.z = field_one(); R.infinity = 0;
            } else {
                R = jacobian_add_mixed(R, pt);
            }
        }
        int w2 = get_nibble_4bit(k2, w);
        if (w2) {
            AffinePoint pt = GENERATOR_TABLE_NIBBLE_PHI[w2];
            if (k2_neg) pt.y = field_negate(pt.y);
            if (R.infinity != 0) {
                R.x = pt.x; R.y = pt.y; R.z = field_one(); R.infinity = 0;
            } else {
                R = jacobian_add_mixed(R, pt);
            }
        }
    }
    return R;
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
    // Phase 4: output_point = hash * G (GLV + precomputed tables)
    // ------------------------------------------------------------------
    Scalar256 hs = scalar_from_bytes(hash);
    JacobianPoint out_jac = scalar_mul_generator_glv_const(hs);
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

// =============================================================================
// Fused BIP-352 kernel with LUT -- entire per-row pipeline in one thread
// =============================================================================

kernel void bip352_fused_kernel_lut(
    device const uchar *tweak_xy     [[buffer(0)]],
    constant uchar *scan_key         [[buffer(1)]],
    device uchar *out_x              [[buffer(2)]],
    device uchar *out_y              [[buffer(3)]],
    constant uint *tag_midstate      [[buffer(4)]],
    constant uint &count             [[buffer(5)]],
    device const AffinePoint *gen_lut [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    // ------------------------------------------------------------------
    // Phase 0: Load inputs (LE wire format -> internal representation)
    // ------------------------------------------------------------------
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
    // Phase 4: output_point = hash * G (LUT: 15 additions, 0 doublings)
    // ------------------------------------------------------------------
    Scalar256 hs = scalar_from_bytes(hash);
    JacobianPoint out_jac = scalar_mul_generator_lut(hs, gen_lut);
    AffinePoint out_aff = jacobian_to_affine(out_jac);

    // ------------------------------------------------------------------
    // Phase 5: Write output as LE bytes
    // ------------------------------------------------------------------
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
