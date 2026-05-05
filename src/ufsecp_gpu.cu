// ============================================================================
// ufsecp_gpu.cu — Fused BIP-352 GPU kernel for Silent Payments scanning
// ============================================================================
// Full pipeline: phases 1-6 on GPU, returning only match flags.
//   Phase 1: shared_secret = tweak_key × scan_key    (predecomp GLV)
//   Phase 2: serialized = SEC1_compressed(shared_secret) || 0x00000000
//   Phase 3: hash = tagged_SHA256("BIP0352/SharedSecret", serialized)
//   Phase 4: output_point = hash × G                 (generator LUT)
//   Phase 5: candidate = output_point + spend_key    (jacobian_add_mixed)
//   Phase 6: prefix match against output list         (point_prefix64)
//
// Fallback path returns affine (x,y) for CPU phases 5-6.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "secp256k1.cuh"
#include "ecdsa.cuh"
#include "affine_add.cuh"
#include "batch_inversion.cuh"

using namespace secp256k1::cuda;

// ============================================================================
// BIP0352/SharedSecret tagged hash midstate
// ============================================================================

__device__ __constant__ static uint32_t BIP352_MIDSTATE[8] = {
    0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
    0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU
};

// ============================================================================
// Predecomposed scan key — per-batch (passed by value to kernels). Living in
// a process-global __constant__ would be racy under concurrent scans with
// different scan keys, so the bytes travel with the per-batch state instead.
// 262 bytes; bundled with BIP352SpendKeys (1092 B) it fits in the 4 KB
// kernel-parameter limit on sm_70+ with room to spare.
// ============================================================================

struct BIP352ScanKeyWnaf {
    int8_t wnaf1[130];
    int8_t wnaf2[130];
    uint8_t k1_neg;
    uint8_t flip_phi;
};

// ============================================================================
// Spend key + label keys — per-batch (passed by value to kernels). Living in
// a process-global __constant__ would be racy under concurrent scans with
// different keys, so the bytes travel with the per-batch state instead.
// 1092 bytes, well under the 4 KB kernel-parameter limit on sm_70+.
// ============================================================================

static constexpr int MAX_LABEL_KEYS = 16;

struct BIP352SpendKeys {
    AffinePoint base;                    // spend_public_key (affine x,y)
    AffinePoint labels[MAX_LABEL_KEYS];  // labelled spend keys
    uint8_t num_labels;
};

// ============================================================================
// Predecomp scalar multiply
// ============================================================================

static constexpr int WNAF_TABLE_SIZE = 8;

__device__ inline void scalar_mul_predecomp(
    const JacobianPoint* p, JacobianPoint* r,
    const BIP352ScanKeyWnaf& scan_wnaf)
{
    AffinePoint base;
    base.x = p->x;
    base.y = p->y;

    if (scan_wnaf.k1_neg) {
        field_negate(&base.y, &base.y);
    }

    AffinePoint tbl_P[WNAF_TABLE_SIZE];
    FieldElement globalz;
    build_wnaf_table_zr(&base, tbl_P, WNAF_TABLE_SIZE, &globalz);

    AffinePoint tbl_phiP[WNAF_TABLE_SIZE];
    derive_endo_table(tbl_P, tbl_phiP, WNAF_TABLE_SIZE, scan_wnaf.flip_phi != 0);

    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    #pragma unroll 1
    for (int i = 129; i >= 0; --i) {
        if (!r->infinity) jacobian_double_unchecked(r, r);

        int8_t d1 = scan_wnaf.wnaf1[i];
        if (d1 != 0) {
            int idx = (((d1 > 0) ? d1 : -d1) - 1) >> 1;
            AffinePoint pt = tbl_P[idx];
            if (d1 < 0) field_negate(&pt.y, &pt.y);
            if (r->infinity) {
                r->x = pt.x; r->y = pt.y;
                field_set_one(&r->z); r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }

        int8_t d2 = scan_wnaf.wnaf2[i];
        if (d2 != 0) {
            int idx = (((d2 > 0) ? d2 : -d2) - 1) >> 1;
            AffinePoint pt = tbl_phiP[idx];
            if (d2 < 0) field_negate(&pt.y, &pt.y);
            if (r->infinity) {
                r->x = pt.x; r->y = pt.y;
                field_set_one(&r->z); r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, &pt, r);
            }
        }
    }

    if (!r->infinity) {
        FieldElement tmp;
        field_mul(&r->z, &globalz, &tmp);
        r->z = tmp;
    }
}

// ============================================================================
// Generator LUT
// ============================================================================

#ifndef LUT_WBITS
#define LUT_WBITS 12
#endif

static constexpr int GEN_LUT_N = (1 << LUT_WBITS);
static constexpr int GEN_LUT_SLICES = (256 + LUT_WBITS - 1) / LUT_WBITS;
static constexpr int GEN_LUT_TOTAL = GEN_LUT_SLICES * GEN_LUT_N;

static constexpr int MAX_CUDA_DEVICES = 16;
static AffinePoint* g_gen_lut[MAX_CUDA_DEVICES] = {};
static bool g_lut_built[MAX_CUDA_DEVICES] = {};
static std::mutex g_lut_mutex;

__device__ inline void scalar_mul_gen_lut(
    const Scalar* k, const AffinePoint* __restrict__ lut, JacobianPoint* r)
{
    r->infinity = true;
    field_set_zero(&r->x);
    field_set_one(&r->y);
    field_set_zero(&r->z);

    constexpr uint32_t MASK = (1u << LUT_WBITS) - 1;

    #pragma unroll 1
    for (int win = 0; win < GEN_LUT_SLICES; win++) {
        int bitpos = win * LUT_WBITS;
        int limb = bitpos >> 6;
        int shift = bitpos & 63;

        uint32_t idx = (uint32_t)((k->limbs[limb] >> shift) & MASK);
        if (shift + LUT_WBITS > 64 && limb < 3)
            idx |= (uint32_t)((k->limbs[limb + 1] << (64 - shift)) & MASK);

        if (idx != 0) {
            const AffinePoint* pt = &lut[(uint32_t)win * GEN_LUT_N + idx];
            if (r->infinity) {
                r->x = pt->x; r->y = pt->y;
                field_set_one(&r->z); r->infinity = false;
            } else {
                jacobian_add_mixed_unchecked(r, pt, r);
            }
        }
    }
}

// ============================================================================
// point_prefix64: extract upper 64 bits of affine x from Jacobian point
// ============================================================================

__device__ inline int64_t point_prefix64(const JacobianPoint* p) {
    if (p->infinity) return 0;

    FieldElement z_inv, z_inv2, ax;
    field_inv(&p->z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&p->x, &z_inv2, &ax);

    uint8_t x_bytes[32];
    field_to_bytes(&ax, x_bytes);

    int64_t prefix = 0;
    for (int i = 0; i < 8; i++)
        prefix = (prefix << 8) | (int64_t)x_bytes[i];
    return prefix;
}

// ============================================================================
// LUT build kernels
// ============================================================================

__global__ void ComputeLutBasePoints(AffinePoint* bases) {
    bases[0] = GENERATOR_TABLE_W8[1];

    JacobianPoint p;
    p.x = GENERATOR_TABLE_W8[1].x;
    p.y = GENERATOR_TABLE_W8[1].y;
    field_set_one(&p.z);
    p.infinity = false;

    for (int i = 1; i < GEN_LUT_SLICES; i++) {
        for (int d = 0; d < LUT_WBITS; d++)
            jacobian_double(&p, &p);

        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p.z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p.x, &z_inv2, &bases[i].x);
        field_mul(&p.y, &z_inv3, &bases[i].y);

        p.x = bases[i].x;
        p.y = bases[i].y;
        field_set_one(&p.z);
    }
}

__global__ void GenLutBuildAffineKernel(
    const AffinePoint* __restrict__ bases,
    AffinePoint* __restrict__ aff_table,
    FieldElement* __restrict__ h_buf,
    int n_entries)
{
    int slice = blockIdx.x;
    if (slice >= GEN_LUT_SLICES) return;

    int offset = slice * n_entries;
    FieldElement* h = h_buf + (size_t)slice * n_entries;

    field_set_zero(&aff_table[offset].x);
    field_set_zero(&aff_table[offset].y);
    aff_table[offset + 1] = bases[slice];

    JacobianPoint acc;
    acc.x = bases[slice].x;
    acc.y = bases[slice].y;
    field_set_one(&acc.z);
    acc.infinity = false;

    for (int j = 2; j < n_entries; j++) {
        FieldElement h_val;
        jacobian_add_mixed_h(&acc, &bases[slice], &acc, h_val);
        h[j - 2] = h_val;
        aff_table[offset + j].x = acc.x;
        aff_table[offset + j].y = acc.y;
    }

    FieldElement z_inv;
    field_inv(&acc.z, &z_inv);

    for (int j = n_entries - 1; j >= 2; --j) {
        FieldElement h_save;
        if (j > 2) h_save = h[j - 2];
        h[j - 2] = z_inv;
        if (j > 2) {
            FieldElement tmp;
            field_mul(&h_save, &z_inv, &tmp);
            z_inv = tmp;
        }
    }
}

__global__ void GenLutConvertZinvKernel(
    AffinePoint* __restrict__ aff_table,
    const FieldElement* __restrict__ h_buf,
    int n_entries)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int per_slice = n_entries - 2;
    int total = GEN_LUT_SLICES * per_slice;
    if (gid >= total) return;

    int slice = gid / per_slice;
    int j = (gid % per_slice) + 2;
    int offset = slice * n_entries;
    const FieldElement* h = h_buf + (size_t)slice * n_entries;

    FieldElement zi = h[j - 2];
    FieldElement z_inv2, z_inv3;
    field_sqr(&zi, &z_inv2);
    field_mul(&zi, &z_inv2, &z_inv3);

    FieldElement ax, ay;
    field_mul(&aff_table[offset + j].x, &z_inv2, &ax);
    field_mul(&aff_table[offset + j].y, &z_inv3, &ay);
    aff_table[offset + j].x = ax;
    aff_table[offset + j].y = ay;
}

static void EnsureGenLutBuilt(int device_id) {
    if (device_id < 0 || device_id >= MAX_CUDA_DEVICES) return;
    if (g_lut_built[device_id]) return;
    std::lock_guard<std::mutex> lock(g_lut_mutex);
    if (g_lut_built[device_id]) return;

    cudaSetDevice(device_id);
    AffinePoint* d_lut = nullptr;

    AffinePoint* d_bases = nullptr;
    if (cudaMalloc(&d_bases, GEN_LUT_SLICES * sizeof(AffinePoint)) != cudaSuccess) {
        g_lut_built[device_id] = true; return;
    }
    ComputeLutBasePoints<<<1, 1>>>(d_bases);
    cudaDeviceSynchronize();

    if (cudaMalloc(&d_lut, (size_t)GEN_LUT_TOTAL * sizeof(AffinePoint)) != cudaSuccess) {
        cudaFree(d_bases); g_lut_built[device_id] = true; return;
    }
    FieldElement* d_h_buf = nullptr;
    if (cudaMalloc(&d_h_buf, (size_t)GEN_LUT_TOTAL * sizeof(FieldElement)) != cudaSuccess) {
        cudaFree(d_bases); cudaFree(d_lut); g_lut_built[device_id] = true; return;
    }

    GenLutBuildAffineKernel<<<GEN_LUT_SLICES, 1>>>(d_bases, d_lut, d_h_buf, GEN_LUT_N);
    cudaDeviceSynchronize();

    int conv_total = GEN_LUT_SLICES * (GEN_LUT_N - 2);
    int conv_blocks = (conv_total + 255) / 256;
    GenLutConvertZinvKernel<<<conv_blocks, 256>>>(d_lut, d_h_buf, GEN_LUT_N);
    cudaDeviceSynchronize();

    cudaFree(d_h_buf);
    cudaFree(d_bases);

    g_gen_lut[device_id] = d_lut;
    g_lut_built[device_id] = true;
    fprintf(stderr, "[CUDA] Generator LUT built on device %d (%d MB)\n",
            device_id, (int)(GEN_LUT_TOTAL * sizeof(AffinePoint) / (1024 * 1024)));
}

// ============================================================================
// Per-batch GPU state
// ============================================================================

struct UfsecpGpuBatchState {
    uint8_t* d_tweak_xy;
    // Full pipeline buffers
    int64_t* d_output_prefixes;
    uint32_t* d_output_offsets;
    uint8_t* d_output_lengths;
    uint8_t* d_match_flags;
    FieldElement* d_cand_x;     // candidate.x (base spend key)
    FieldElement* d_cand_z;     // candidate.z → batch inverted in-place
    JacobianPoint* d_output_pts; // output points for label processing
    // Fallback buffers (phases 1-4 only)
    uint8_t* d_output_x;
    uint8_t* d_output_y;
    // Per-batch scan-key wNAF plan + spend keys (passed by value to kernels)
    BIP352ScanKeyWnaf h_scan_wnaf;
    BIP352SpendKeys h_spend_keys;
    uint32_t count;
    cudaStream_t stream;
    int device_id;
    bool full_pipeline;
};

// ============================================================================
// Shared: phases 1-3 (identical for all kernel variants)
// ============================================================================

__device__ inline void bip352_phases_1_3(
    const uint8_t* tweak, JacobianPoint* shared_point,
    uint8_t serialized[37],
    const BIP352ScanKeyWnaf& scan_wnaf)
{
    FieldElement fx, fy;
    for (int i = 0; i < 4; i++) {
        uint64_t lx = 0, ly = 0;
        for (int j = 7; j >= 0; j--) {
            lx = (lx << 8) | tweak[i * 8 + j];
            ly = (ly << 8) | tweak[32 + i * 8 + j];
        }
        fx.limbs[i] = lx;
        fy.limbs[i] = ly;
    }

    JacobianPoint tweak_point;
    tweak_point.x = fx;
    tweak_point.y = fy;
    field_set_one(&tweak_point.z);
    tweak_point.infinity = false;

    // Phase 1
    scalar_mul_predecomp(&tweak_point, shared_point, scan_wnaf);

    // Phase 2
    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv(&shared_point->z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&shared_point->x, &z_inv2, &x_aff);
    field_mul(&shared_point->y, &z_inv3, &y_aff);

    uint8_t shared_x[32];
    field_to_bytes(&x_aff, shared_x);
    uint8_t prefix = field_is_odd(&y_aff) ? 0x03 : 0x02;

    serialized[0] = prefix;
    for (int i = 0; i < 32; i++)
        serialized[1 + i] = shared_x[i];
    serialized[33] = 0; serialized[34] = 0;
    serialized[35] = 0; serialized[36] = 0;
}

__device__ inline void bip352_phase_3_hash(const uint8_t serialized[37], uint8_t hash[32]) {
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++)
        ctx.h[i] = BIP352_MIDSTATE[i];
    ctx.buf_len = 0;
    ctx.total = 64;
    sha256_update(&ctx, serialized, 37);
    sha256_final(&ctx, hash);
}

// ============================================================================
// Full pipeline: three-pass with GPU batch inversion
// ============================================================================
// Pass 1: phases 1-4 + jacobian_add_mixed(output, spend_key) → store candidate
//         Extract candidate.z into separate buffer for batch inversion
// Pass 2: batch_inverse_kernel on candidate Z values (1 field_inv per 256 threads)
// Pass 3: use precomputed Z^{-1} to extract prefix cheaply (2 muls) + match
//         Repeat for labels using stored output_point (per-thread field_inv for labels)

// Pass 1: phases 1-4, add spend key, store candidate X/Z + output point for labels
__global__ void BIP352FullPass1(
    const uint8_t* __restrict__ tweak_xy,
    const AffinePoint* __restrict__ gen_lut,
    FieldElement* __restrict__ cand_x,      // candidate.x for base spend key
    FieldElement* __restrict__ cand_z,      // candidate.z → batch inverted in pass 2
    JacobianPoint* __restrict__ output_pts, // output points for label processing (may be null)
    BIP352ScanKeyWnaf scan_wnaf,            // per-batch (by value) — see header note
    BIP352SpendKeys spend_keys,             // per-batch (by value) — see header note
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    JacobianPoint shared_point;
    uint8_t serialized[37];
    bip352_phases_1_3(tweak_xy + idx * 64, &shared_point, serialized, scan_wnaf);

    uint8_t hash[32];
    bip352_phase_3_hash(serialized, hash);

    Scalar hash_scalar;
    scalar_from_bytes(hash, &hash_scalar);
    JacobianPoint output_point;
    scalar_mul_gen_lut(&hash_scalar, gen_lut, &output_point);

    // Phase 5: candidate = output_point + spend_key
    JacobianPoint candidate;
    jacobian_add_mixed_unchecked(&output_point, &spend_keys.base, &candidate);

    cand_x[idx] = candidate.x;
    cand_z[idx] = candidate.z;

    // Store output point for label processing (if labels are present)
    if (output_pts) output_pts[idx] = output_point;
}

// Helper: in-block batch inversion using shared memory prefix/suffix scans.
// Computes z_inv[tid] = input[tid]^{-1} for all valid threads in the block.
// Uses 1 field_inv per block (amortized over blockDim.x threads).
// L and R are shared memory arrays of size >= blockDim.x.
__device__ inline void block_batch_invert(
    const FieldElement& input, FieldElement* z_inv_out,
    FieldElement* L, FieldElement* R, int valid_in_block)
{
    int tid = threadIdx.x;

    L[tid] = input;
    R[tid] = input;
    __syncthreads();

    block_prefix_mul(L, L, valid_in_block);
    block_suffix_mul(R, R, valid_in_block);

    __shared__ FieldElement total_inv;
    if (tid == 0 && valid_in_block > 0) {
        FieldElement total_prod = L[valid_in_block - 1];
        field_inv(&total_prod, &total_inv);
    }
    __syncthreads();

    FieldElement z_inv = total_inv;
    if (tid > 0)
        field_mul(&z_inv, &L[tid - 1], &z_inv);
    if (tid < valid_in_block - 1)
        field_mul(&z_inv, &R[tid + 1], &z_inv);

    *z_inv_out = z_inv;
    __syncthreads();
}

// Helper: extract upper 64 bits of affine x from Jacobian X and Z^{-1}.
__device__ inline int64_t extract_prefix(const FieldElement& jac_x, const FieldElement& z_inv) {
    FieldElement z_inv2, ax;
    field_sqr(&z_inv, &z_inv2);
    field_mul(&jac_x, &z_inv2, &ax);

    uint8_t x_bytes[32];
    field_to_bytes(&ax, x_bytes);
    int64_t pfx = 0;
    for (int i = 0; i < 8; i++)
        pfx = (pfx << 8) | (int64_t)x_bytes[i];
    return pfx;
}

// Fused batch inversion + prefix extraction + matching.
// Base case: batch-invert candidate Z (from Pass 1).
// Labels: batch-invert label candidate Z within the same kernel (1 round per label).
// Zero per-thread field_inv for any code path.
__global__ __launch_bounds__(256, 4) void BIP352BatchInvMatchKernel(
    const FieldElement* __restrict__ cand_x,
    const FieldElement* __restrict__ cand_z,
    const JacobianPoint* __restrict__ output_pts,
    const int64_t* __restrict__ output_prefixes,
    const uint32_t* __restrict__ output_offsets,
    const uint8_t* __restrict__ output_lengths,
    uint8_t* __restrict__ match_flags,
    BIP352SpendKeys spend_keys,             // per-batch (by value)
    uint32_t count)
{
    using namespace secp256k1::cuda;

    extern __shared__ FieldElement shared_mem[];
    FieldElement* L = shared_mem;
    FieldElement* R = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    bool valid = (idx < (int)count);

    int valid_in_block = (int)count - blockIdx.x * blockDim.x;
    if (valid_in_block > (int)blockDim.x) valid_in_block = (int)blockDim.x;

    uint32_t off = 0;
    uint8_t len = 0;
    bool found = false;

    if (valid) {
        off = output_offsets[idx];
        len = output_lengths[idx];
    }

    // === Round 1: base spend key (batch-invert candidate Z) ===
    {
        FieldElement z_val;
        if (valid) z_val = cand_z[idx];
        else { z_val.limbs[0] = 1; z_val.limbs[1] = 0; z_val.limbs[2] = 0; z_val.limbs[3] = 0; }

        FieldElement z_inv;
        block_batch_invert(z_val, &z_inv, L, R, valid_in_block);

        if (valid) {
            int64_t pfx = extract_prefix(cand_x[idx], z_inv);
            for (uint8_t j = 0; j < len && !found; j++)
                if (output_prefixes[off + j] == pfx) found = true;
        }
    }

    // === Rounds 2+: label keys (batch-invert label candidate Z) ===
    if (output_pts) {
        for (uint8_t lbl = 0; lbl < spend_keys.num_labels; lbl++) {
            // Compute label candidate and extract its Z
            FieldElement label_z, label_x;
            if (valid) {
                JacobianPoint label_cand;
                jacobian_add_mixed_unchecked(&output_pts[idx], &spend_keys.labels[lbl], &label_cand);
                label_x = label_cand.x;
                label_z = label_cand.z;
            } else {
                label_z.limbs[0] = 1; label_z.limbs[1] = 0;
                label_z.limbs[2] = 0; label_z.limbs[3] = 0;
            }

            FieldElement label_z_inv;
            block_batch_invert(label_z, &label_z_inv, L, R, valid_in_block);

            if (valid && !found) {
                int64_t label_pfx = extract_prefix(label_x, label_z_inv);
                for (uint8_t j = 0; j < len && !found; j++)
                    if (output_prefixes[off + j] == label_pfx) found = true;
            }
        }
    }

    if (valid)
        match_flags[idx] = found ? 1 : 0;
}

// ============================================================================
// Fallback kernels: phases 1-4, return affine (x,y) for CPU phases 5-6
// ============================================================================

__global__ void BIP352FusedKernelLUT(
    const uint8_t* __restrict__ tweak_xy,
    uint8_t* __restrict__ out_x,
    uint8_t* __restrict__ out_y,
    const AffinePoint* __restrict__ gen_lut,
    BIP352ScanKeyWnaf scan_wnaf,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    JacobianPoint shared_point;
    uint8_t serialized[37];
    bip352_phases_1_3(tweak_xy + idx * 64, &shared_point, serialized, scan_wnaf);

    uint8_t hash[32];
    bip352_phase_3_hash(serialized, hash);

    Scalar hash_scalar;
    scalar_from_bytes(hash, &hash_scalar);
    JacobianPoint output_point;
    scalar_mul_gen_lut(&hash_scalar, gen_lut, &output_point);

    jacobian_to_affine(&output_point.x, &output_point.y, &output_point.z);

    uint8_t ox_be[32], oy_be[32];
    field_to_bytes(&output_point.x, ox_be);
    field_to_bytes(&output_point.y, oy_be);

    uint8_t* dst_x = out_x + idx * 32;
    uint8_t* dst_y = out_y + idx * 32;
    for (int i = 0; i < 32; i++) {
        dst_x[i] = ox_be[31 - i];
        dst_y[i] = oy_be[31 - i];
    }
}

__global__ void BIP352FusedKernel(
    const uint8_t* __restrict__ tweak_xy,
    uint8_t* __restrict__ out_x,
    uint8_t* __restrict__ out_y,
    BIP352ScanKeyWnaf scan_wnaf,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    JacobianPoint shared_point;
    uint8_t serialized[37];
    bip352_phases_1_3(tweak_xy + idx * 64, &shared_point, serialized, scan_wnaf);

    uint8_t hash[32];
    bip352_phase_3_hash(serialized, hash);

    Scalar hash_scalar;
    scalar_from_bytes(hash, &hash_scalar);
    JacobianPoint output_point;
    scalar_mul_generator_const(&hash_scalar, &output_point);

    jacobian_to_affine(&output_point.x, &output_point.y, &output_point.z);

    uint8_t ox_be[32], oy_be[32];
    field_to_bytes(&output_point.x, ox_be);
    field_to_bytes(&output_point.y, oy_be);

    uint8_t* dst_x = out_x + idx * 32;
    uint8_t* dst_y = out_y + idx * 32;
    for (int i = 0; i < 32; i++) {
        dst_x[i] = ox_be[31 - i];
        dst_y[i] = oy_be[31 - i];
    }
}

// ============================================================================
// Extern "C" interface
// ============================================================================

extern "C" {

int UfsecpCudaDetect(int* num_gpus) {
    int total = 0;
    cudaError_t err = cudaGetDeviceCount(&total);
    if (err != cudaSuccess) { *num_gpus = 0; return -1; }

    // Kernels are compiled for sm_80+ (Ampere). If any visible device is older,
    // report 0 so the caller falls through to OpenCL — otherwise launches on
    // the unsupported device would fail with cudaErrorNoKernelImageForDevice.
    for (int i = 0; i < total; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess || prop.major < 8) {
            *num_gpus = 0;
            return 0;
        }
    }
    *num_gpus = total;
    return 0;
}

// Convert 64-byte LE spend/label bytes (Frigate wire format) into the host
// BIP352SpendKeys layout that the kernel expects (per-batch).
static void BuildSpendKeys(const uint8_t* spend_xy, int num_labels,
                           const uint8_t* label_keys_xy,
                           BIP352SpendKeys& out) {
    out = {};

    // spend_xy: 64 LE bytes (32 x + 32 y) → AffinePoint (4×u64 LE limbs)
    for (int i = 0; i < 4; i++) {
        uint64_t xv = 0, yv = 0;
        for (int j = 0; j < 8; j++) {
            xv |= (uint64_t)spend_xy[i * 8 + j] << (j * 8);
            yv |= (uint64_t)spend_xy[32 + i * 8 + j] << (j * 8);
        }
        out.base.x.limbs[i] = xv;
        out.base.y.limbs[i] = yv;
    }

    int n = (num_labels > MAX_LABEL_KEYS) ? MAX_LABEL_KEYS : num_labels;
    out.num_labels = (uint8_t)n;
    for (int L = 0; L < n; L++) {
        const uint8_t* lxy = label_keys_xy + L * 64;
        for (int i = 0; i < 4; i++) {
            uint64_t xv = 0, yv = 0;
            for (int j = 0; j < 8; j++) {
                xv |= (uint64_t)lxy[i * 8 + j] << (j * 8);
                yv |= (uint64_t)lxy[32 + i * 8 + j] << (j * 8);
            }
            out.labels[L].x.limbs[i] = xv;
            out.labels[L].y.limbs[i] = yv;
        }
    }
}

// Full pipeline launch: upload tweaks + output prefix data + per-batch spend
void* UfsecpCudaLaunchBatchFull(
    const uint8_t* tweak_data,
    const int64_t* output_prefixes, uint32_t total_outputs,
    const uint32_t* output_offsets,
    const uint8_t* output_lengths,
    uint32_t count, int device_id,
    const void* precomp,
    const uint8_t* spend_xy, int num_labels,
    const uint8_t* label_keys_xy)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return nullptr;

    auto* state = new UfsecpGpuBatchState();
    state->count = count;
    state->device_id = device_id;
    state->full_pipeline = true;
    if (precomp)
        std::memcpy(&state->h_scan_wnaf, precomp, sizeof(BIP352ScanKeyWnaf));
    else
        std::memset(&state->h_scan_wnaf, 0, sizeof(BIP352ScanKeyWnaf));
    state->d_tweak_xy = nullptr;
    state->d_output_prefixes = nullptr;
    state->d_output_offsets = nullptr;
    state->d_output_lengths = nullptr;
    state->d_match_flags = nullptr;
    state->d_cand_x = nullptr;
    state->d_cand_z = nullptr;
    state->d_output_pts = nullptr;
    state->d_output_x = nullptr;
    state->d_output_y = nullptr;

    // Per-batch spend keys (no global, so concurrent scans with different keys
    // don't race on a shared __constant__).
    BuildSpendKeys(spend_xy, num_labels, label_keys_xy, state->h_spend_keys);

    err = cudaStreamCreate(&state->stream);
    if (err != cudaSuccess) { delete state; return nullptr; }

    size_t tweak_size = (size_t)count * 64;

    err = cudaMalloc(&state->d_tweak_xy, tweak_size);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&state->d_output_prefixes, (size_t)total_outputs * sizeof(int64_t));
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&state->d_output_offsets, (size_t)count * sizeof(uint32_t));
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&state->d_output_lengths, (size_t)count);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&state->d_match_flags, (size_t)count);
    if (err != cudaSuccess) goto fail;

    // Intermediate buffers for batch inversion (candidate X and Z)
    err = cudaMalloc(&state->d_cand_x, (size_t)count * sizeof(FieldElement));
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&state->d_cand_z, (size_t)count * sizeof(FieldElement));
    if (err != cudaSuccess) goto fail;

    // Output points for label processing (needed when labels > 0)
    err = cudaMalloc(&state->d_output_pts, (size_t)count * sizeof(JacobianPoint));
    if (err != cudaSuccess) goto fail;

    cudaMemcpyAsync(state->d_tweak_xy, tweak_data, tweak_size,
                    cudaMemcpyHostToDevice, state->stream);
    cudaMemcpyAsync(state->d_output_prefixes, output_prefixes,
                    (size_t)total_outputs * sizeof(int64_t),
                    cudaMemcpyHostToDevice, state->stream);
    cudaMemcpyAsync(state->d_output_offsets, output_offsets,
                    (size_t)count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, state->stream);
    cudaMemcpyAsync(state->d_output_lengths, output_lengths,
                    (size_t)count,
                    cudaMemcpyHostToDevice, state->stream);

    return state;

fail:
    if (state->d_tweak_xy) cudaFree(state->d_tweak_xy);
    if (state->d_output_prefixes) cudaFree(state->d_output_prefixes);
    if (state->d_output_offsets) cudaFree(state->d_output_offsets);
    if (state->d_output_lengths) cudaFree(state->d_output_lengths);
    if (state->d_match_flags) cudaFree(state->d_match_flags);
    if (state->d_cand_x) cudaFree(state->d_cand_x);
    if (state->d_cand_z) cudaFree(state->d_cand_z);
    if (state->d_output_pts) cudaFree(state->d_output_pts);
    cudaStreamDestroy(state->stream);
    delete state;
    return nullptr;
}

int UfsecpCudaRunKernelsFull(void* state_handle, uint8_t* match_flags, uint32_t count) {
    auto* state = static_cast<UfsecpGpuBatchState*>(state_handle);
    if (!state) return -1;

    cudaSetDevice(state->device_id);
    EnsureGenLutBuilt(state->device_id);

    AffinePoint* lut = g_gen_lut[state->device_id];
    if (!lut) return -1;  // full pipeline requires LUT

    int threads = 128;
    int blocks = ((int)count + threads - 1) / threads;

    // Pass 1: phases 1-4 + add spend key → store candidate X/Z + output points
    BIP352FullPass1<<<blocks, threads, 0, state->stream>>>(
        state->d_tweak_xy, lut,
        state->d_cand_x, state->d_cand_z, state->d_output_pts,
        state->h_scan_wnaf, state->h_spend_keys, count);

    // Pass 2: fused batch inversion + prefix extraction + matching
    {
        int bi_threads = 256;
        int bi_blocks = ((int)count + bi_threads - 1) / bi_threads;
        size_t shared_size = 2 * bi_threads * sizeof(FieldElement);
        BIP352BatchInvMatchKernel<<<bi_blocks, bi_threads, shared_size, state->stream>>>(
            state->d_cand_x, state->d_cand_z, state->d_output_pts,
            state->d_output_prefixes, state->d_output_offsets, state->d_output_lengths,
            state->d_match_flags, state->h_spend_keys, count);
    }

    cudaMemcpyAsync(match_flags, state->d_match_flags, (size_t)count,
                    cudaMemcpyDeviceToHost, state->stream);

    cudaStreamSynchronize(state->stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] RunKernelsFull error: %s\n", cudaGetErrorString(err));
    }
    return (err == cudaSuccess) ? 0 : -1;
}

// Legacy interface (phases 1-4 only, for fallback)
void* UfsecpCudaLaunchBatch(
    const uint8_t* scan_key, const uint8_t* tweak_data,
    uint32_t count, int device_id, const void* precomp)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return nullptr;

    auto* state = new UfsecpGpuBatchState();
    state->count = count;
    state->device_id = device_id;
    state->full_pipeline = false;
    if (precomp)
        std::memcpy(&state->h_scan_wnaf, precomp, sizeof(BIP352ScanKeyWnaf));
    else
        std::memset(&state->h_scan_wnaf, 0, sizeof(BIP352ScanKeyWnaf));
    state->d_tweak_xy = nullptr;
    state->d_output_x = nullptr;
    state->d_output_y = nullptr;
    state->d_output_prefixes = nullptr;
    state->d_output_offsets = nullptr;
    state->d_output_lengths = nullptr;
    state->d_match_flags = nullptr;
    state->d_cand_x = nullptr;
    state->d_cand_z = nullptr;
    state->d_output_pts = nullptr;

    err = cudaStreamCreate(&state->stream);
    if (err != cudaSuccess) { delete state; return nullptr; }

    size_t tweak_size = (size_t)count * 64;
    size_t point_size = (size_t)count * 32;

    err = cudaMalloc(&state->d_tweak_xy, tweak_size);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&state->d_output_x, point_size);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(&state->d_output_y, point_size);
    if (err != cudaSuccess) goto fail;

    cudaMemcpyAsync(state->d_tweak_xy, tweak_data, tweak_size,
                    cudaMemcpyHostToDevice, state->stream);

    return state;

fail:
    if (state->d_tweak_xy) cudaFree(state->d_tweak_xy);
    if (state->d_output_x) cudaFree(state->d_output_x);
    if (state->d_output_y) cudaFree(state->d_output_y);
    cudaStreamDestroy(state->stream);
    delete state;
    return nullptr;
}

int UfsecpCudaRunKernels(void* state_handle, uint8_t* out_x, uint8_t* out_y, uint32_t count) {
    auto* state = static_cast<UfsecpGpuBatchState*>(state_handle);
    if (!state) return -1;

    cudaSetDevice(state->device_id);
    EnsureGenLutBuilt(state->device_id);

    int threads = 128;
    int blocks = ((int)count + threads - 1) / threads;

    AffinePoint* lut = g_gen_lut[state->device_id];
    if (lut) {
        BIP352FusedKernelLUT<<<blocks, threads, 0, state->stream>>>(
            state->d_tweak_xy, state->d_output_x, state->d_output_y,
            lut, state->h_scan_wnaf, count);
    } else {
        BIP352FusedKernel<<<blocks, threads, 0, state->stream>>>(
            state->d_tweak_xy, state->d_output_x, state->d_output_y,
            state->h_scan_wnaf, count);
    }

    size_t point_size = (size_t)count * 32;
    cudaMemcpyAsync(out_x, state->d_output_x, point_size,
                    cudaMemcpyDeviceToHost, state->stream);
    cudaMemcpyAsync(out_y, state->d_output_y, point_size,
                    cudaMemcpyDeviceToHost, state->stream);

    cudaStreamSynchronize(state->stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] RunKernels (legacy) error: %s\n", cudaGetErrorString(err));
    }
    return (err == cudaSuccess) ? 0 : -1;
}

void UfsecpCudaFreeBatch(void* state_handle) {
    if (!state_handle) return;
    auto* state = static_cast<UfsecpGpuBatchState*>(state_handle);

    cudaSetDevice(state->device_id);
    cudaStreamSynchronize(state->stream);

    if (state->d_tweak_xy) cudaFree(state->d_tweak_xy);
    if (state->d_output_x) cudaFree(state->d_output_x);
    if (state->d_output_y) cudaFree(state->d_output_y);
    if (state->d_output_prefixes) cudaFree(state->d_output_prefixes);
    if (state->d_output_offsets) cudaFree(state->d_output_offsets);
    if (state->d_output_lengths) cudaFree(state->d_output_lengths);
    if (state->d_match_flags) cudaFree(state->d_match_flags);
    if (state->d_cand_x) cudaFree(state->d_cand_x);
    if (state->d_cand_z) cudaFree(state->d_cand_z);
    if (state->d_output_pts) cudaFree(state->d_output_pts);

    cudaStreamDestroy(state->stream);
    delete state;
}

} // extern "C"
