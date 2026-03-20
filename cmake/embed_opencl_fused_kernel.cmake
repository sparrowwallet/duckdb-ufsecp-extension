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
    "${KERNEL_DIR}/secp256k1_bip352.cl"
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

# Part 2: Extension kernels (fused pipeline + LUT build)
#
# The fused kernels use scalar_mul_glv_predecomp_impl from secp256k1_bip352.cl
# which reads precomputed wNAF digits from __constant memory (BIP352ScanKeyGlv),
# and bip352_shared_secret_input_impl / bip352_tagged_sha256_impl for phases 2-3.

set(EXT_KERNELS "
// =============================================================================
// Generator LUT lookup (w=12: 22 slices x 4096 entries = 5 MB)
// =============================================================================

#define LUT_WBITS       12
#define GEN_LUT_N       (1 << LUT_WBITS)
#define GEN_LUT_SLICES  ((256 + LUT_WBITS - 1) / LUT_WBITS)

inline void scalar_mul_gen_lut(JacobianPoint* r, const Scalar* k,
                               __global const AffinePoint* lut) {
    point_set_infinity(r);
    uint mask = (1u << LUT_WBITS) - 1;
    for (int win = 0; win < GEN_LUT_SLICES; win++) {
        int bitpos = win * LUT_WBITS;
        int limb = bitpos >> 6;
        int shift = bitpos & 63;
        uint idx = (uint)((k->limbs[limb] >> shift) & mask);
        if (shift + LUT_WBITS > 64 && limb < 3)
            idx |= (uint)((k->limbs[limb + 1] << (64 - shift)) & mask);
        if (idx != 0) {
            AffinePoint pt = lut[(uint)win * GEN_LUT_N + idx];
            if (point_is_infinity(r)) {
                point_from_affine(r, &pt);
            } else {
                point_add_mixed_impl(r, r, &pt);
            }
        }
    }
}

// =============================================================================
// Fused BIP-352 kernel -- predecomp scan key + GLV generator multiplication
// =============================================================================

__kernel void bip352_fused_kernel(
    __global const uchar *tweak_xy,
    __constant const BIP352ScanKeyGlv *scan_plan,
    __global uchar *out_x,
    __global uchar *out_y,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Load tweak point from LE wire format
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

    // Phase 1: shared_secret = scan_key * tweak_point (predecomposed GLV)
    JacobianPoint shared_jac;
    scalar_mul_glv_predecomp_impl(&shared_jac, &tweak_pt, scan_plan);

    // Phase 2: serialize to SEC1 compressed + 4 zero bytes
    uchar ser[37];
    bip352_shared_secret_input_impl(&shared_jac, ser);

    // Phase 3: tagged SHA-256
    uchar hash[32];
    bip352_tagged_sha256_impl(ser, 37, hash);

    // Phase 4: output_point = hash * G (GLV windowed)
    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);
    JacobianPoint out_jac;
    scalar_mul_generator_windowed_impl(&out_jac, &hs);

    AffinePoint out_aff;
    jacobian_to_affine_convert_impl(&out_aff,
        &out_jac.x, &out_jac.y, &out_jac.z);

    // Phase 5: write output as LE bytes
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
    __constant const BIP352ScanKeyGlv *scan_plan,
    __global uchar *out_x,
    __global uchar *out_y,
    __global const AffinePoint *gen_lut,
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

    JacobianPoint shared_jac;
    scalar_mul_glv_predecomp_impl(&shared_jac, &tweak_pt, scan_plan);

    uchar ser[37];
    bip352_shared_secret_input_impl(&shared_jac, ser);

    uchar hash[32];
    bip352_tagged_sha256_impl(ser, 37, hash);

    // Phase 4: k*G via LUT (w=12, 21 additions, 0 doublings)
    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);
    JacobianPoint out_jac;
    scalar_mul_gen_lut(&out_jac, &hs, gen_lut);

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
// Generator LUT build kernels
// =============================================================================

__kernel void compute_lut_base_points(__global AffinePoint* bases) {
    AffinePoint g_local = GENERATOR_TABLE_W8[1];
    bases[0] = g_local;

    JacobianPoint p;
    point_from_affine(&p, &g_local);

    for (int i = 1; i < GEN_LUT_SLICES; i++) {
        for (int d = 0; d < LUT_WBITS; d++)
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

    aff_table[offset].x.limbs[0] = 0; aff_table[offset].x.limbs[1] = 0;
    aff_table[offset].x.limbs[2] = 0; aff_table[offset].x.limbs[3] = 0;
    aff_table[offset].y.limbs[0] = 0; aff_table[offset].y.limbs[1] = 0;
    aff_table[offset].y.limbs[2] = 0; aff_table[offset].y.limbs[3] = 0;

    aff_table[offset + 1] = bases[slice];

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

    FieldElement z_inv;
    field_inv_impl(&z_inv, &acc.z);

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

    FieldElement jx = aff_table[offset + j].x;
    FieldElement jy = aff_table[offset + j].y;
    FieldElement ax, ay;
    field_mul_impl(&ax, &jx, &z_inv2);
    field_mul_impl(&ay, &jy, &z_inv3);
    aff_table[offset + j].x = ax;
    aff_table[offset + j].y = ay;
}

// =============================================================================
// Full pipeline: phases 1-6 on GPU with batch inversion
// =============================================================================

#define MAX_LABEL_KEYS 16

typedef struct {
    AffinePoint base;
    AffinePoint labels[MAX_LABEL_KEYS];
    uchar num_labels;
    uchar pad[3];
} BIP352SpendKeys;

// Hillis-Steele parallel prefix multiply (inclusive, local memory)
inline void ocl_block_prefix_mul(__local FieldElement* data, int n) {
    int tid = get_local_id(0);
    for (int offset = 1; offset < n; offset *= 2) {
        FieldElement val;
        if (tid >= offset && tid < n) {
            FieldElement a = data[tid - offset], b = data[tid];
            field_mul_impl(&val, &a, &b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid >= offset && tid < n)
            data[tid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Hillis-Steele parallel suffix multiply (inclusive, local memory)
inline void ocl_block_suffix_mul(__local FieldElement* data, int n) {
    int tid = get_local_id(0);
    for (int offset = 1; offset < n; offset *= 2) {
        FieldElement val;
        if (tid + offset < n) {
            FieldElement a = data[tid + offset], b = data[tid];
            field_mul_impl(&val, &a, &b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid + offset < n)
            data[tid] = val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

// Helper: in-block batch inversion using local memory prefix/suffix scans.
// ocl_block_batch_invert: broadcasts total inverse via R[0] (not needed for recovery)
inline void ocl_block_batch_invert(
    FieldElement input, FieldElement* z_inv_out,
    __local FieldElement* L, __local FieldElement* R, int valid_in_block)
{
    int tid = get_local_id(0);
    L[tid] = input;
    R[tid] = input;
    barrier(CLK_LOCAL_MEM_FENCE);

    ocl_block_prefix_mul(L, valid_in_block);
    ocl_block_suffix_mul(R, valid_in_block);

    // Thread 0 computes total inverse and broadcasts via R[0]
    // (R[0] is not read by any thread in the recovery step — only R[tid+1] is used)
    if (tid == 0 && valid_in_block > 0) {
        FieldElement total_prod = L[valid_in_block - 1];
        FieldElement inv_result;
        field_inv_impl(&inv_result, &total_prod);
        R[0] = inv_result;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    FieldElement total_inv = R[0];

    FieldElement z_inv = total_inv;
    if (tid > 0) {
        FieldElement lprev = L[tid - 1];
        field_mul_impl(&z_inv, &z_inv, &lprev);
    }
    if (tid < valid_in_block - 1) {
        FieldElement rnext = R[tid + 1];
        field_mul_impl(&z_inv, &z_inv, &rnext);
    }

    *z_inv_out = z_inv;
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Helper: extract upper 64 bits of affine x from Jacobian X and Z^{-1}.
inline long ocl_extract_prefix(const FieldElement* jac_x, const FieldElement* z_inv) {
    FieldElement z_inv2, ax;
    field_sqr_impl(&z_inv2, z_inv);
    field_mul_impl(&ax, jac_x, &z_inv2);

    uchar x_bytes[32];
    field_to_bytes_impl(&ax, x_bytes);
    long pfx = 0;
    for (int i = 0; i < 8; i++)
        pfx = (pfx << 8) | (long)x_bytes[i];
    return pfx;
}

// Pass 1: phases 1-4, add spend key, store candidate X/Z + output points
__kernel void bip352_full_pass1(
    __global const uchar *tweak_xy,
    __constant const BIP352ScanKeyGlv *scan_plan,
    __global const AffinePoint *gen_lut,
    __constant const BIP352SpendKeys *spend,
    __global FieldElement *cand_x,
    __global FieldElement *cand_z,
    __global JacobianPoint *output_pts,
    const uint count)
{
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

    JacobianPoint shared_jac;
    scalar_mul_glv_predecomp_impl(&shared_jac, &tweak_pt, scan_plan);

    uchar ser[37];
    bip352_shared_secret_input_impl(&shared_jac, ser);

    uchar hash[32];
    bip352_tagged_sha256_impl(ser, 37, hash);

    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);
    JacobianPoint output_point;
    scalar_mul_gen_lut(&output_point, &hs, gen_lut);

    // Store output point for label processing
    output_pts[gid] = output_point;

    // Add spend key
    AffinePoint spend_base = spend->base;
    JacobianPoint candidate;
    point_add_mixed_impl(&candidate, &output_point, &spend_base);

    cand_x[gid] = candidate.x;
    cand_z[gid] = candidate.z;
}

// Fused batch inversion + prefix extraction + matching.
// Base case: batch-invert candidate Z. Labels: batch-invert per label round.
__kernel void bip352_batch_inv_match(
    __global const FieldElement *cand_x,
    __global const FieldElement *cand_z,
    __global const JacobianPoint *output_pts,
    __constant const BIP352SpendKeys *spend,
    __global const long *output_prefixes,
    __global const uint *output_offsets,
    __global const uchar *output_lengths,
    __global uchar *match_flags,
    __local FieldElement *shared_mem,
    const uint count)
{
    int tid = get_local_id(0);
    int gid = get_global_id(0);
    int valid = (gid < (int)count);

    int valid_in_block = (int)count - (int)get_group_id(0) * (int)get_local_size(0);
    if (valid_in_block > (int)get_local_size(0)) valid_in_block = (int)get_local_size(0);

    __local FieldElement *L = shared_mem;
    __local FieldElement *R = shared_mem + get_local_size(0);

    uint off = 0;
    uchar len = 0;
    int found = 0;

    if (valid) {
        off = output_offsets[gid];
        len = output_lengths[gid];
    }

    // Round 1: base spend key
    {
        FieldElement z_val;
        if (valid) z_val = cand_z[gid];
        else { z_val.limbs[0] = 1UL; z_val.limbs[1] = 0; z_val.limbs[2] = 0; z_val.limbs[3] = 0; }

        FieldElement z_inv;
        ocl_block_batch_invert(z_val, &z_inv, L, R, valid_in_block);

        if (valid) {
            FieldElement cx = cand_x[gid];
            long pfx = ocl_extract_prefix(&cx, &z_inv);
            for (uchar j = 0; j < len && !found; j++)
                if (output_prefixes[off + j] == pfx) found = 1;
        }
    }

    // Rounds 2+: label keys
    for (uchar lbl = 0; lbl < spend->num_labels; lbl++) {
        FieldElement label_z, label_x;
        if (valid) {
            JacobianPoint op = output_pts[gid];
            AffinePoint label_key = spend->labels[lbl];
            JacobianPoint label_cand;
            point_add_mixed_impl(&label_cand, &op, &label_key);
            label_x = label_cand.x;
            label_z = label_cand.z;
        } else {
            label_z.limbs[0] = 1UL; label_z.limbs[1] = 0;
            label_z.limbs[2] = 0; label_z.limbs[3] = 0;
        }

        FieldElement label_z_inv;
        ocl_block_batch_invert(label_z, &label_z_inv, L, R, valid_in_block);

        if (valid && !found) {
            long label_pfx = ocl_extract_prefix(&label_x, &label_z_inv);
            for (uchar j = 0; j < len && !found; j++)
                if (output_prefixes[off + j] == label_pfx) found = 1;
        }
    }

    if (valid)
        match_flags[gid] = found ? 1 : 0;
}
")

# Write as C++ header with multiple string literals to stay under MSVC's 65535-byte limit.
# Adjacent raw string literals are concatenated by the compiler.

file(WRITE "${OUTPUT_FILE}" "// Auto-generated by cmake/embed_opencl_fused_kernel.cmake -- do not edit\n")
file(APPEND "${OUTPUT_FILE}" "#pragma once\n\n")
file(APPEND "${OUTPUT_FILE}" "static const char OPENCL_FUSED_KERNEL_SOURCE[] =\n")
file(APPEND "${OUTPUT_FILE}" "R\"opencl_p1(\n")
file(APPEND "${OUTPUT_FILE}" "${UF_KERNELS}")
file(APPEND "${OUTPUT_FILE}" ")opencl_p1\"\n")
file(APPEND "${OUTPUT_FILE}" "R\"opencl_p2(\n")
file(APPEND "${OUTPUT_FILE}" "${EXT_KERNELS}")
file(APPEND "${OUTPUT_FILE}" ")opencl_p2\";\n")
