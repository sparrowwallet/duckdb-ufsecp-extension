# embed_opencl_fused_kernel.cmake -- Generate opencl_fused_kernel_source.h
#
# Inputs (passed via -D):
#   KERNEL_DIR  -- Path to UltrafastSecp256k1/opencl/kernels/
#   OUTPUT_FILE -- Path to write opencl_fused_kernel_source.h
#
# Reads the required kernel files, strips #include and #pragma once lines,
# and appends the fused BIP-352 kernels + LUT build kernels.
# Writes the result as a C++ raw string literal.

set(KERNEL_FILES
    "${KERNEL_DIR}/secp256k1_field.cl"
    "${KERNEL_DIR}/secp256k1_point.cl"
    "${KERNEL_DIR}/secp256k1_gen_table_w8.cl"
    "${KERNEL_DIR}/secp256k1_extended.cl"
    "${KERNEL_DIR}/secp256k1_affine.cl"
)

# Part 1: UltrafastSecp256k1 kernel files (stripped of includes/guards)
set(UF_KERNELS "")

foreach(FILE ${KERNEL_FILES})
    file(READ "${FILE}" CONTENT)
    string(REGEX REPLACE "#include [^\n]*\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#pragma once[^\n]*\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#ifndef SECP256K1_[A-Z_]*_CL\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#define SECP256K1_[A-Z_]*_CL\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#endif // SECP256K1_[A-Z_]*_CL\n" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "#endif [^\n]*SECP256K1_[A-Z_]*_CL[^\n]*\n" "" CONTENT "${CONTENT}")
    string(APPEND UF_KERNELS "${CONTENT}\n")
endforeach()

# Part 2: Extension kernels (precomputed tables, GLV helpers, fused kernels, LUT build)
set(EXT_KERNELS "
// =============================================================================
// Precomputed generator nibble tables for GLV generator multiplication
// =============================================================================
// GENERATOR_TABLE_NIBBLE[i] = i*G  (i=0..15)
// GENERATOR_TABLE_NIBBLE_PHI[i] = phi(i*G) where phi is the GLV endomorphism

__constant AffinePoint GENERATOR_TABLE_NIBBLE[16] = {
    {{{0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL}},{{0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL}}},
    {{{0x59f2815b16f81798UL,0x029bfcdb2dce28d9UL,0x55a06295ce870b07UL,0x79be667ef9dcbbacUL}},{{0x9c47d08ffb10d4b8UL,0xfd17b448a6855419UL,0x5da4fbfc0e1108a8UL,0x483ada7726a3c465UL}}},
    {{{0xabac09b95c709ee5UL,0x5c778e4b8cef3ca7UL,0x3045406e95c07cd8UL,0xc6047f9441ed7d6dUL}},{{0x236431a950cfe52aUL,0xf7f632653266d0e1UL,0xa3c58419466ceaeeUL,0x1ae168fea63dc339UL}}},
    {{{0x8601f113bce036f9UL,0xb531c845836f99b0UL,0x49344f85f89d5229UL,0xf9308a019258c310UL}},{{0x6cb9fd7584b8e672UL,0x6500a99934c2231bUL,0x0fe337e62a37f356UL,0x388f7b0f632de814UL}}},
    {{{0x74fa94abe8c4cd13UL,0xcc6c13900ee07584UL,0x581e4904930b1404UL,0xe493dbf1c10d80f3UL}},{{0xcfe97bdc47739922UL,0xd967ae33bfbdfe40UL,0x5642e2098ea51448UL,0x51ed993ea0d455b7UL}}},
    {{{0xcba8d569b240efe4UL,0xe88b84bddc619ab7UL,0x55b4a7250a5c5128UL,0x2f8bde4d1a072093UL}},{{0xdca87d3aa6ac62d6UL,0xf788271bab0d6840UL,0xd4dba9dda6c9c426UL,0xd8ac222636e5e3d6UL}}},
    {{{0x2f057a1460297556UL,0x82f6472f8568a18bUL,0x20453a14355235d3UL,0xfff97bd5755eeea4UL}},{{0x3c870c36b075f297UL,0xde80f0f6518fe4a0UL,0xf3be96017f45c560UL,0xae12777aacfbb620UL}}},
    {{{0xe92bddedcac4f9bcUL,0x3d419b7e0330e39cUL,0xa398f365f2ea7a0eUL,0x5cbdf0646e5db4eaUL}},{{0xa5082628087264daUL,0xa813d0b813fde7b5UL,0xa3178d6d861a54dbUL,0x6aebca40ba255960UL}}},
    {{{0x67784ef3e10a2a01UL,0x0a1bdd05e5af888aUL,0xaff3843fb70f3c2fUL,0x2f01e5e15cca351dUL}},{{0xb5da2cb76cbde904UL,0xc2e213d6ba5b7617UL,0x293d082a132d13b4UL,0x5c4da8a741539949UL}}},
    {{{0xc35f110dfc27ccbeUL,0xe09796974c57e714UL,0x09ad178a9f559abdUL,0xacd484e2f0c7f653UL}},{{0x05cc262ac64f9c37UL,0xadd888a4375f8e0fUL,0x64380971763b61e9UL,0xcc338921b0a7d9fdUL}}},
    {{{0x52a68e2a47e247c7UL,0x3442d49b1943c2b7UL,0x35477c7b1ae6ae5dUL,0xa0434d9e47f3c862UL}},{{0x3cbee53b037368d7UL,0x6f794c2ed877a159UL,0xa3b6c7e693a24c69UL,0x893aba425419bc27UL}}},
    {{{0xbbec17895da008cbUL,0x5649980be5c17891UL,0x5ef4246b70c65aacUL,0x774ae7f858a9411eUL}},{{0x301d74c9c953c61bUL,0x372db1e2dff9d6a8UL,0x0243dd56d7b7b365UL,0xd984a032eb6b5e19UL}}},
    {{{0xc5b0f47070afe85aUL,0x687cf4419620095bUL,0x15c38f004d734633UL,0xd01115d548e7561bUL}},{{0x6b051b13f4062327UL,0x79238c5dd9a86d52UL,0xa8b64537e17bd815UL,0xa9f34ffdc815e0d7UL}}},
    {{{0xdeeddf8f19405aa8UL,0xb075fbc6610e58cdUL,0xc7d1d205c3748651UL,0xf28773c2d975288bUL}},{{0x29b5cb52db03ed81UL,0x3a1a06da521fa91fUL,0x758212eb65cdaf47UL,0x0ab0902e8d880a89UL}}},
    {{{0xe49b241a60e823e4UL,0x26aa7b63678949e6UL,0xfd64e67f07d38e32UL,0x499fdf9e895e719cUL}},{{0xc65f40d403a13f5bUL,0x464279c27a3f95bcUL,0x90f044e4a7b3d464UL,0xcac2f6c4b54e8551UL}}},
    {{{0x44adbcf8e27e080eUL,0x31e5946f3c85f79eUL,0x5a465ae3095ff411UL,0xd7924d4f7d43ea96UL}},{{0xc504dc9ff6a26b58UL,0xea40af2bd896d3a5UL,0x83842ec228cc6defUL,0x581e2872a86c72a6UL}}}
};

__constant AffinePoint GENERATOR_TABLE_NIBBLE_PHI[16] = {
    {{{0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL}},{{0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL,0x0000000000000000UL}}},
    {{{0xa7bba04400b88fcbUL,0x872844067f15e98dUL,0xab0102b696902325UL,0xbcace2e99da01887UL}},{{0x9c47d08ffb10d4b8UL,0xfd17b448a6855419UL,0x5da4fbfc0e1108a8UL,0x483ada7726a3c465UL}}},
    {{{0x3e995b6ed89250e1UL,0xd2fad8cce43837efUL,0x4135ee7d59f87b33UL,0xc360a6d0b34ce6dfUL}},{{0x236431a950cfe52aUL,0xf7f632653266d0e1UL,0xa3c58419466ceaeeUL,0x1ae168fea63dc339UL}}},
    {{{0xf7f0728c77206b2fUL,0x8af1e022c6dc8e1cUL,0x8dcd8dcf2a28fa2fUL,0xdf6edf03731f9b4bUL}},{{0x6cb9fd7584b8e672UL,0x6500a99934c2231bUL,0x0fe337e62a37f356UL,0x388f7b0f632de814UL}}},
    {{{0x5bde5b333b306100UL,0x714c30b5ab487127UL,0x5c45faf8b90e324bUL,0x1b77921f0d382907UL}},{{0xcfe97bdc47739922UL,0xd967ae33bfbdfe40UL,0x5642e2098ea51448UL,0x51ed993ea0d455b7UL}}},
    {{{0x138c694695a83668UL,0xa045693ee0d097ccUL,0xf79f54fbccb94671UL,0x337b52e3acda49dfUL}},{{0xdca87d3aa6ac62d6UL,0xf788271bab0d6840UL,0xd4dba9dda6c9c426UL,0xd8ac222636e5e3d6UL}}},
    {{{0x47aaf28078f38045UL,0x86649d3e56a15a68UL,0x5e3aa731e3e8bed7UL,0xe63bcdd9aa535fc6UL}},{{0x3c870c36b075f297UL,0xde80f0f6518fe4a0UL,0xf3be96017f45c560UL,0xae12777aacfbb620UL}}},
    {{{0x3bc4686e4e53bc94UL,0x0d3b20e20faf7aaaUL,0xa4fec4d1c095c06eUL,0x13f26e754bea0b77UL}},{{0xa5082628087264daUL,0xa813d0b813fde7b5UL,0xa3178d6d861a54dbUL,0x6aebca40ba255960UL}}},
    {{{0x03e947742446cc73UL,0xb4ff771524257657UL,0xaa77840f29e24892UL,0x47ab650342d401a7UL}},{{0xb5da2cb76cbde904UL,0xc2e213d6ba5b7617UL,0x293d082a132d13b4UL,0x5c4da8a741539949UL}}},
    {{{0x20cd912e65953a52UL,0xb565cdf5ef6d44e1UL,0x7b6558afec58ab20UL,0x87b404037e44e819UL}},{{0x05cc262ac64f9c37UL,0xadd888a4375f8e0fUL,0x64380971763b61e9UL,0xcc338921b0a7d9fdUL}}},
    {{{0xbdb3e957741afe29UL,0xc1938d8e083762e4UL,0xa136ebb246813990UL,0x26ce269bf7a397b1UL}},{{0x3cbee53b037368d7UL,0x6f794c2ed877a159UL,0xa3b6c7e693a24c69UL,0x893aba425419bc27UL}}},
    {{{0xc5ff4334bb209ce7UL,0x79859bb70b5ff620UL,0x8d897c41bebf1a26UL,0x51f4d3d1171dac1dUL}},{{0x301d74c9c953c61bUL,0x372db1e2dff9d6a8UL,0x0243dd56d7b7b365UL,0xd984a032eb6b5e19UL}}},
    {{{0x4a3eb52c042295e5UL,0xf9482837c9535355UL,0xac1548422eac82adUL,0x88591bfd953aac41UL}},{{0x6b051b13f4062327UL,0x79238c5dd9a86d52UL,0xa8b64537e17bd815UL,0xa9f34ffdc815e0d7UL}}},
    {{{0x60aaee6a475fb678UL,0x32907ed74a3d0562UL,0x07046c4578fc783bUL,0xf14d58374bb890a2UL}},{{0x29b5cb52db03ed81UL,0x3a1a06da521fa91fUL,0x758212eb65cdaf47UL,0x0ab0902e8d880a89UL}}},
    {{{0x0e6ab7ee20a0b458UL,0x580656a627c529f6UL,0x1548f0dc87c37384UL,0x7b1252177810048aUL}},{{0xc65f40d403a13f5bUL,0x464279c27a3f95bcUL,0x90f044e4a7b3d464UL,0xcac2f6c4b54e8551UL}}},
    {{{0x3ac0a40c71b1b3b4UL,0x05cc3bc9c1c0a639UL,0x0e1b4825512b6948UL,0x805f1105f5f9454aUL}},{{0xc504dc9ff6a26b58UL,0xea40af2bd896d3a5UL,0x83842ec228cc6defUL,0x581e2872a86c72a6UL}}}
};

// =============================================================================
// Helpers for GLV generator multiplication
// =============================================================================

inline int get_window_4bit_fused(const Scalar* s, int pos) {
    int bp = pos * 4, li = bp >> 6, sh = bp & 63;
    ulong v = s->limbs[li] >> sh;
    if (sh > 60 && li < 3) v |= s->limbs[li+1] << (64 - sh);
    return (int)(v & 0xFUL);
}

// GLV generator multiplication using precomputed __constant nibble tables.
// Replaces scalar_mul_generator_windowed_impl which builds tables at runtime.
inline void scalar_mul_generator_glv(JacobianPoint* r, const Scalar* k) {
    if ((k->limbs[0]|k->limbs[1]|k->limbs[2]|k->limbs[3]) == 0) {
        point_set_infinity(r);
        return;
    }

    Scalar k1, k2; int k1_neg, k2_neg;
    glv_decompose_impl(k, &k1, &k2, &k1_neg, &k2_neg);

    int bl1 = scalar_bitlen_impl(&k1);
    int bl2 = scalar_bitlen_impl(&k2);
    int max_bits = (bl1 > bl2) ? bl1 : bl2;
    int num_windows = (max_bits + 3) / 4;

    point_set_infinity(r);
    for (int w = num_windows - 1; w >= 0; --w) {
        if (!point_is_infinity(r)) {
            point_double_impl(r, r); point_double_impl(r, r);
            point_double_impl(r, r); point_double_impl(r, r);
        }
        int w1 = get_window_4bit_fused(&k1, w);
        if (w1) {
            AffinePoint pt = GENERATOR_TABLE_NIBBLE[w1];
            if (k1_neg) field_neg_impl(&pt.y, &pt.y);
            point_add_mixed_impl(r, r, &pt);
        }
        int w2 = get_window_4bit_fused(&k2, w);
        if (w2) {
            AffinePoint pt = GENERATOR_TABLE_NIBBLE_PHI[w2];
            if (k2_neg) field_neg_impl(&pt.y, &pt.y);
            point_add_mixed_impl(r, r, &pt);
        }
    }
}

// =============================================================================
// Fused BIP-352 kernel -- entire per-row pipeline in one thread (GLV fallback)
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

    uchar sk_be[32];
    for (int i = 0; i < 32; i++) sk_be[i] = scan_key[31 - i];
    Scalar sk;
    scalar_from_bytes_impl(sk_be, &sk);

    JacobianPoint shared_jac;
    scalar_mul_glv_impl(&shared_jac, &sk, &tweak_pt);

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

    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = tag_midstate[i];
    ctx.buf_len = 0;
    ctx.total_len = 64;
    sha256_update(&ctx, ser, 37);
    uchar hash[32];
    sha256_final(&ctx, hash);

    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);
    JacobianPoint out_jac;
    scalar_mul_generator_glv(&out_jac, &hs);

    AffinePoint out_aff;
    jacobian_to_affine_convert_impl(&out_aff,
        &out_jac.x, &out_jac.y, &out_jac.z);

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

// =============================================================================
// LUT-accelerated fused BIP-352 kernel
// =============================================================================

__kernel void bip352_fused_kernel_lut(
    __global const uchar *tweak_xy,
    __constant uchar *scan_key,
    __global uchar *out_x,
    __global uchar *out_y,
    __constant uint *tag_midstate,
    __global const AffinePoint *gen_lut,   // 16 x 65536 AffinePoints
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

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

    uchar sk_be[32];
    for (int i = 0; i < 32; i++) sk_be[i] = scan_key[31 - i];
    Scalar sk;
    scalar_from_bytes_impl(sk_be, &sk);

    JacobianPoint shared_jac;
    scalar_mul_glv_impl(&shared_jac, &sk, &tweak_pt);

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

    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = tag_midstate[i];
    ctx.buf_len = 0;
    ctx.total_len = 64;
    sha256_update(&ctx, ser, 37);
    uchar hash[32];
    sha256_final(&ctx, hash);

    // Phase 4: k*G via LUT (15 additions, 0 doublings)
    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);
    JacobianPoint out_jac;
    scalar_mul_generator_lut_impl(&out_jac, &hs, gen_lut);

    AffinePoint out_aff;
    jacobian_to_affine_convert_impl(&out_aff,
        &out_jac.x, &out_jac.y, &out_jac.z);

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

// =============================================================================
// Generator LUT build kernels (16 x 65536 = 64 MB precomputed table)
// =============================================================================

#define GEN_LUT_SLICES  16
#define GEN_LUT_N       65536

// Single work-item: compute B_i = 2^(16*i) * G for i=0..15
__kernel void compute_lut_base_points(__global AffinePoint* bases) {
    // Copy G from __constant to private memory
    AffinePoint g_local = GENERATOR_TABLE_W8[1];
    bases[0] = g_local;

    JacobianPoint p;
    point_from_affine(&p, &g_local);

    for (int i = 1; i < GEN_LUT_SLICES; i++) {
        for (int d = 0; d < 16; d++)
            point_double_impl(&p, &p);

        FieldElement z_inv, z_inv2, z_inv3, ax, ay;
        field_inv_impl(&z_inv, &p.z);
        field_sqr_impl(&z_inv2, &z_inv);
        field_mul_impl(&z_inv3, &z_inv, &z_inv2);
        field_mul_impl(&ax, &p.x, &z_inv2);
        field_mul_impl(&ay, &p.y, &z_inv3);
        bases[i].x = ax;
        bases[i].y = ay;

        p.x = ax;
        p.y = ay;
        p.z.limbs[0] = 1UL; p.z.limbs[1] = 0; p.z.limbs[2] = 0; p.z.limbs[3] = 0;
        p.infinity = 0;
    }
}

// Fused LUT build + serial inversion (one work-item per slice).
__kernel void gen_lut_build_affine_kernel(
    __global const AffinePoint* bases,
    __global AffinePoint* aff_table,
    __global FieldElement* h_buf,
    const int n_entries)
{
    int slice = get_global_id(0);
    if (slice >= GEN_LUT_SLICES) return;

    int offset = slice * n_entries;
    __global FieldElement* h = h_buf + (long)slice * n_entries;

    // [0] = identity
    aff_table[offset].x.limbs[0] = 0; aff_table[offset].x.limbs[1] = 0;
    aff_table[offset].x.limbs[2] = 0; aff_table[offset].x.limbs[3] = 0;
    aff_table[offset].y.limbs[0] = 0; aff_table[offset].y.limbs[1] = 0;
    aff_table[offset].y.limbs[2] = 0; aff_table[offset].y.limbs[3] = 0;

    // [1] = base point
    aff_table[offset + 1] = bases[slice];

    // Forward pass
    AffinePoint base_pt = bases[slice];
    JacobianPoint acc;
    point_from_affine(&acc, &base_pt);

    for (int j = 2; j < n_entries; j++) {
        FieldElement h_val;
        point_add_mixed_h_impl(&acc, &acc, &base_pt, &h_val);
        h[j - 2] = h_val;
        aff_table[offset + j].x = acc.x;
        aff_table[offset + j].y = acc.y;
    }

    // Single inversion of final Z
    FieldElement z_inv;
    field_inv_impl(&z_inv, &acc.z);

    // Backward sweep
    for (int j = n_entries - 1; j >= 2; --j) {
        FieldElement h_save;
        if (j > 2) h_save = h[j - 2];
        h[j - 2] = z_inv;
        if (j > 2) {
            FieldElement tmp;
            field_mul_impl(&tmp, &h_save, &z_inv);
            z_inv = tmp;
        }
    }
}

// Parallel affine conversion using precomputed Z^{-1} from h_buf.
__kernel void gen_lut_convert_zinv_kernel(
    __global AffinePoint* aff_table,
    __global const FieldElement* h_buf,
    const int n_entries)
{
    int gid = get_global_id(0);
    int per_slice = n_entries - 2;
    int total = GEN_LUT_SLICES * per_slice;
    if (gid >= total) return;

    int slice = gid / per_slice;
    int j = (gid % per_slice) + 2;
    int offset = slice * n_entries;
    __global const FieldElement* h = h_buf + (long)slice * n_entries;

    FieldElement zi = h[j - 2];
    FieldElement z_inv2, z_inv3;
    field_sqr_impl(&z_inv2, &zi);
    field_mul_impl(&z_inv3, &zi, &z_inv2);

    // Copy from __global to private for field_mul_impl
    FieldElement jx = aff_table[offset + j].x;
    FieldElement jy = aff_table[offset + j].y;
    FieldElement ax, ay;
    field_mul_impl(&ax, &jx, &z_inv2);
    field_mul_impl(&ay, &jy, &z_inv3);
    aff_table[offset + j].x = ax;
    aff_table[offset + j].y = ay;
}
")

# Write as C++ header with two string literals to stay under MSVC's 65535-byte limit.
# Adjacent raw string literals are concatenated by the compiler.

# Split: UF kernel files go in part 1, extension kernels (tables + fused) go in part 2.
file(WRITE "${OUTPUT_FILE}" "// Auto-generated by cmake/embed_opencl_fused_kernel.cmake -- do not edit\n")
file(APPEND "${OUTPUT_FILE}" "#pragma once\n\n")
file(APPEND "${OUTPUT_FILE}" "static const char OPENCL_FUSED_KERNEL_SOURCE[] =\n")
file(APPEND "${OUTPUT_FILE}" "R\"opencl_p1(\n")
file(APPEND "${OUTPUT_FILE}" "${UF_KERNELS}")
file(APPEND "${OUTPUT_FILE}" ")opencl_p1\"\n")
file(APPEND "${OUTPUT_FILE}" "R\"opencl_p2(\n")
file(APPEND "${OUTPUT_FILE}" "${EXT_KERNELS}")
file(APPEND "${OUTPUT_FILE}" ")opencl_p2\";\n")
