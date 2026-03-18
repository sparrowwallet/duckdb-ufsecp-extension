// ============================================================================
// ufsecp_gpu.cu — Fused BIP-352 GPU kernel for Silent Payments scanning
// ============================================================================
// This file implements the GPU-accelerated phases 1-4 of the BIP-352
// scanning pipeline using UltrafastSecp256k1's CUDA device functions.
//
// Each CUDA thread processes one transaction independently:
//   Phase 1: shared_secret = tweak_key × scan_key    (scalar_mul_glv_wnaf)
//   Phase 2: serialized = SEC1_compressed(shared_secret) || 0x00000000
//   Phase 3: hash = tagged_SHA256("BIP0352/SharedSecret", serialized)
//   Phase 4: output_point = hash × G                 (scalar_mul_generator_lut)
//
// Phase 4 uses a precomputed 64 MB generator LUT (16 windows × 65536 entries)
// for 15 mixed additions with zero doublings, replacing the w=4 fixed-window
// approach (252 doublings + ~64 additions). Falls back to scalar_mul_generator_const
// if LUT allocation fails.
//
// Phases 5-6 (batch affine add + match) run on CPU.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

// UltrafastSecp256k1 CUDA headers
#include "secp256k1.cuh"    // FieldElement, Scalar, JacobianPoint, AffinePoint,
                            // scalar_mul_glv_wnaf, scalar_mul_generator_const,
                            // scalar_mul_generator_lut, jacobian_add_mixed_h,
                            // field_inv/sqr/mul, field_set_one, field_is_odd,
                            // GENERATOR_TABLE_W8
#include "ecdsa.cuh"        // scalar_from_bytes, field_to_bytes, SHA256Ctx, sha256_*
#include "affine_add.cuh"   // jacobian_to_affine

using namespace secp256k1::cuda;

// ============================================================================
// BIP0352/SharedSecret tagged hash midstate
// ============================================================================

__device__ __constant__ static uint32_t BIP352_MIDSTATE[8] = {
    0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
    0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU
};

// ============================================================================
// Generator LUT — precomputed table for k*G (built once, persistent)
// ============================================================================
// Table layout: 16 slices × 65536 AffinePoints = 1,048,576 points (64 MB)
// lut[win * 65536 + idx] = idx × (2^(16*win)) × G
// Lookup: split scalar into 16 × 16-bit windows, index each slice, sum.

static constexpr int GEN_LUT_N = 65536;
static constexpr int GEN_LUT_SLICES = 16;
static constexpr int GEN_LUT_TOTAL = GEN_LUT_SLICES * GEN_LUT_N;

static AffinePoint* g_gen_lut = nullptr;
static bool g_lut_built = false;
static std::mutex g_lut_mutex;

// ============================================================================
// LUT build kernels (adapted from bench_bip352.cu)
// ============================================================================

// Single-thread kernel: compute B_i = 2^(16*i) × G for i=0..15
__global__ void ComputeLutBasePoints(AffinePoint* bases) {
    // bases[0] = G
    bases[0] = GENERATOR_TABLE_W8[1];

    JacobianPoint p;
    p.x = GENERATOR_TABLE_W8[1].x;
    p.y = GENERATOR_TABLE_W8[1].y;
    field_set_one(&p.z);
    p.infinity = false;

    for (int i = 1; i < GEN_LUT_SLICES; i++) {
        // Double 16 times: p = 2^16 × previous
        for (int d = 0; d < 16; d++)
            jacobian_double(&p, &p);

        // Convert to affine
        FieldElement z_inv, z_inv2, z_inv3;
        field_inv(&p.z, &z_inv);
        field_sqr(&z_inv, &z_inv2);
        field_mul(&z_inv2, &z_inv, &z_inv3);
        field_mul(&p.x, &z_inv2, &bases[i].x);
        field_mul(&p.y, &z_inv3, &bases[i].y);

        // Reset to affine for next chain
        p.x = bases[i].x;
        p.y = bases[i].y;
        field_set_one(&p.z);
    }
}

// Fused LUT build + serial inversion (one block per slice, 1 thread each).
// Forward pass: sequential jacobian_add_mixed_h, stores H values + Jacobian X,Y.
// Single field_inv of final Z per slice (16 total instead of 1,048,576).
// Backward sweep: reconstructs per-point Z^{-1} into h_buf.
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

    // [0] = identity
    field_set_zero(&aff_table[offset].x);
    field_set_zero(&aff_table[offset].y);

    // [1] = base point (already affine)
    aff_table[offset + 1] = bases[slice];

    // Forward pass: build chain P[j] = P[j-1] + base
    JacobianPoint acc;
    acc.x = bases[slice].x;
    acc.y = bases[slice].y;
    field_set_one(&acc.z);
    acc.infinity = false;

    for (int j = 2; j < n_entries; j++) {
        FieldElement h_val;
        jacobian_add_mixed_h(&acc, &bases[slice], &acc, h_val);
        h[j - 2] = h_val;
        // Store Jacobian X,Y temporarily in affine output buffer
        aff_table[offset + j].x = acc.x;
        aff_table[offset + j].y = acc.y;
    }

    // Single inversion of final Z
    FieldElement z_inv;
    field_inv(&acc.z, &z_inv);

    // Backward sweep: reconstruct per-point Z^{-1}
    // Recurrence: Z_j = Z_{j-1} * H_j  =>  Z_{j-1}^{-1} = H_j * Z_j^{-1}
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

// Parallel affine conversion using precomputed Z^{-1} from h_buf.
// One thread per convertible point (j=2..N-1 in each slice).
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

// ============================================================================
// Lazy LUT construction (called once on first kernel dispatch)
// ============================================================================

static void EnsureGenLutBuilt(int device_id) {
    if (g_lut_built) return;
    std::lock_guard<std::mutex> lock(g_lut_mutex);
    if (g_lut_built) return;

    cudaSetDevice(device_id);

    // Build into a local pointer; only publish to g_gen_lut after
    // all kernels have completed, so concurrent readers never see
    // a non-null pointer to an incomplete table.
    AffinePoint* d_lut = nullptr;

    // Step 1: compute 16 base points
    AffinePoint* d_bases = nullptr;
    if (cudaMalloc(&d_bases, GEN_LUT_SLICES * sizeof(AffinePoint)) != cudaSuccess) {
        g_lut_built = true;  // mark as attempted, g_gen_lut stays null -> fallback
        return;
    }
    ComputeLutBasePoints<<<1, 1>>>(d_bases);
    cudaDeviceSynchronize();

    // Step 2: allocate LUT (64 MB) + temp H buffer (32 MB)
    if (cudaMalloc(&d_lut, (size_t)GEN_LUT_TOTAL * sizeof(AffinePoint)) != cudaSuccess) {
        cudaFree(d_bases);
        g_lut_built = true;
        return;
    }
    FieldElement* d_h_buf = nullptr;
    if (cudaMalloc(&d_h_buf, (size_t)GEN_LUT_TOTAL * sizeof(FieldElement)) != cudaSuccess) {
        cudaFree(d_bases);
        cudaFree(d_lut);
        g_lut_built = true;
        return;
    }

    // Step 3: fused build + serial inversion (16 blocks × 1 thread)
    GenLutBuildAffineKernel<<<GEN_LUT_SLICES, 1>>>(d_bases, d_lut, d_h_buf, GEN_LUT_N);
    cudaDeviceSynchronize();

    // Step 4: parallel affine conversion
    int conv_total = GEN_LUT_SLICES * (GEN_LUT_N - 2);
    int conv_blocks = (conv_total + 255) / 256;
    GenLutConvertZinvKernel<<<conv_blocks, 256>>>(d_lut, d_h_buf, GEN_LUT_N);
    cudaDeviceSynchronize();

    // Cleanup temp buffers
    cudaFree(d_h_buf);
    cudaFree(d_bases);

    // Publish: only now is the table fully populated and safe to read
    g_gen_lut = d_lut;
    g_lut_built = true;
    fprintf(stderr, "[CUDA] Generator LUT built (%d MB)\n",
            (int)(GEN_LUT_TOTAL * sizeof(AffinePoint) / (1024 * 1024)));
}

// ============================================================================
// Per-batch GPU state
// ============================================================================

struct UfsecpGpuBatchState {
    uint8_t* d_tweak_xy;     // Device: N × 64 bytes (32B x LE || 32B y LE)
    uint8_t* d_output_x;     // Device: N × 32 bytes (affine x, LE)
    uint8_t* d_output_y;     // Device: N × 32 bytes (affine y, LE)
    uint8_t* d_scan_key;     // Device: 32 bytes (scan private key, LE)
    uint32_t count;
    cudaStream_t stream;
    int device_id;
};

// ============================================================================
// Fused BIP-352 kernel with LUT — one thread per transaction
// ============================================================================

__global__ void BIP352FusedKernelLUT(
    const uint8_t* __restrict__ tweak_xy,   // N × 64 bytes
    const uint8_t* __restrict__ scan_key,   // 32 bytes (same for all threads)
    uint8_t* __restrict__ out_x,            // N × 32 bytes
    uint8_t* __restrict__ out_y,            // N × 32 bytes
    const AffinePoint* __restrict__ gen_lut, // 16 × 65536 AffinePoints
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // ----------------------------------------------------------------
    // Load inputs (LE wire format → internal representation)
    // ----------------------------------------------------------------
    const uint8_t* tweak = tweak_xy + idx * 64;

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

    uint8_t sk_be[32];
    for (int i = 0; i < 32; i++)
        sk_be[i] = scan_key[31 - i];
    Scalar sk;
    scalar_from_bytes(sk_be, &sk);

    // ----------------------------------------------------------------
    // Phase 1: Shared secret
    // ----------------------------------------------------------------
    JacobianPoint shared_point;
    scalar_mul_glv_wnaf(&tweak_point, &sk, &shared_point);

    // ----------------------------------------------------------------
    // Phase 2: Jacobian → affine → SEC1 compressed serialization
    // ----------------------------------------------------------------
    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv(&shared_point.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&shared_point.x, &z_inv2, &x_aff);
    field_mul(&shared_point.y, &z_inv3, &y_aff);

    uint8_t shared_x[32];
    field_to_bytes(&x_aff, shared_x);
    uint8_t prefix = field_is_odd(&y_aff) ? 0x03 : 0x02;

    uint8_t serialized[37];
    serialized[0] = prefix;
    for (int i = 0; i < 32; i++)
        serialized[1 + i] = shared_x[i];
    serialized[33] = 0; serialized[34] = 0;
    serialized[35] = 0; serialized[36] = 0;

    // ----------------------------------------------------------------
    // Phase 3: Tagged hash with BIP0352/SharedSecret midstate
    // ----------------------------------------------------------------
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++)
        ctx.h[i] = BIP352_MIDSTATE[i];
    ctx.buf_len = 0;
    ctx.total = 64;

    sha256_update(&ctx, serialized, 37);
    uint8_t hash[32];
    sha256_final(&ctx, hash);

    // ----------------------------------------------------------------
    // Phase 4: Generator multiplication via LUT — hash × G
    // ----------------------------------------------------------------
    // 15 mixed additions, 0 doublings (vs 252 doublings + ~64 additions)
    Scalar hash_scalar;
    scalar_from_bytes(hash, &hash_scalar);

    JacobianPoint output_point;
    scalar_mul_generator_lut(&hash_scalar, gen_lut, &output_point);

    // Convert Jacobian → affine
    jacobian_to_affine(&output_point.x, &output_point.y, &output_point.z);

    // ----------------------------------------------------------------
    // Write output: affine (x, y) as LE bytes
    // ----------------------------------------------------------------
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
// Fallback: Fused BIP-352 kernel without LUT (w=4 __constant__ table)
// ============================================================================

__global__ void BIP352FusedKernel(
    const uint8_t* __restrict__ tweak_xy,
    const uint8_t* __restrict__ scan_key,
    uint8_t* __restrict__ out_x,
    uint8_t* __restrict__ out_y,
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* tweak = tweak_xy + idx * 64;

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

    uint8_t sk_be[32];
    for (int i = 0; i < 32; i++)
        sk_be[i] = scan_key[31 - i];
    Scalar sk;
    scalar_from_bytes(sk_be, &sk);

    JacobianPoint shared_point;
    scalar_mul_glv_wnaf(&tweak_point, &sk, &shared_point);

    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv(&shared_point.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&shared_point.x, &z_inv2, &x_aff);
    field_mul(&shared_point.y, &z_inv3, &y_aff);

    uint8_t shared_x[32];
    field_to_bytes(&x_aff, shared_x);
    uint8_t prefix = field_is_odd(&y_aff) ? 0x03 : 0x02;

    uint8_t serialized[37];
    serialized[0] = prefix;
    for (int i = 0; i < 32; i++)
        serialized[1 + i] = shared_x[i];
    serialized[33] = 0; serialized[34] = 0;
    serialized[35] = 0; serialized[36] = 0;

    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++)
        ctx.h[i] = BIP352_MIDSTATE[i];
    ctx.buf_len = 0;
    ctx.total = 64;

    sha256_update(&ctx, serialized, 37);
    uint8_t hash[32];
    sha256_final(&ctx, hash);

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
// Extern "C" interface — called from ufsecp_extension.cpp
// ============================================================================

extern "C" {

int UfsecpCudaDetect(int* num_gpus) {
    cudaError_t err = cudaGetDeviceCount(num_gpus);
    if (err != cudaSuccess) {
        *num_gpus = 0;
        return -1;
    }
    return 0;
}

void* UfsecpCudaLaunchBatch(
    const uint8_t* scan_key,       // 32 bytes (LE, same for all rows)
    const uint8_t* tweak_data,     // N × 64 bytes
    uint32_t count,
    int device_id)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return nullptr;

    auto* state = new UfsecpGpuBatchState();
    state->count = count;
    state->device_id = device_id;
    state->d_tweak_xy = nullptr;
    state->d_output_x = nullptr;
    state->d_output_y = nullptr;
    state->d_scan_key = nullptr;

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

    err = cudaMalloc(&state->d_scan_key, 32);
    if (err != cudaSuccess) goto fail;

    // Copy inputs to device
    cudaMemcpyAsync(state->d_tweak_xy, tweak_data, tweak_size,
                    cudaMemcpyHostToDevice, state->stream);
    cudaMemcpyAsync(state->d_scan_key, scan_key, 32,
                    cudaMemcpyHostToDevice, state->stream);

    return state;

fail:
    if (state->d_tweak_xy) cudaFree(state->d_tweak_xy);
    if (state->d_output_x) cudaFree(state->d_output_x);
    if (state->d_output_y) cudaFree(state->d_output_y);
    if (state->d_scan_key) cudaFree(state->d_scan_key);
    cudaStreamDestroy(state->stream);
    delete state;
    return nullptr;
}

int UfsecpCudaRunKernels(
    void* state_handle,
    uint8_t* out_x,            // host: N × 32 bytes (LE)
    uint8_t* out_y,            // host: N × 32 bytes (LE)
    uint32_t count)
{
    auto* state = static_cast<UfsecpGpuBatchState*>(state_handle);
    if (!state) return -1;

    cudaSetDevice(state->device_id);

    // Build LUT on first use (lazy, thread-safe)
    EnsureGenLutBuilt(state->device_id);

    int threads = 128;
    int blocks = ((int)count + threads - 1) / threads;

    if (g_gen_lut) {
        BIP352FusedKernelLUT<<<blocks, threads, 0, state->stream>>>(
            state->d_tweak_xy,
            state->d_scan_key,
            state->d_output_x,
            state->d_output_y,
            g_gen_lut,
            count
        );
    } else {
        // Fallback: no LUT available (allocation failed)
        BIP352FusedKernel<<<blocks, threads, 0, state->stream>>>(
            state->d_tweak_xy,
            state->d_scan_key,
            state->d_output_x,
            state->d_output_y,
            count
        );
    }

    // Copy results back to host
    size_t point_size = (size_t)count * 32;
    cudaMemcpyAsync(out_x, state->d_output_x, point_size,
                    cudaMemcpyDeviceToHost, state->stream);
    cudaMemcpyAsync(out_y, state->d_output_y, point_size,
                    cudaMemcpyDeviceToHost, state->stream);

    cudaStreamSynchronize(state->stream);

    cudaError_t err = cudaGetLastError();
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
    if (state->d_scan_key) cudaFree(state->d_scan_key);

    cudaStreamDestroy(state->stream);
    delete state;
}

} // extern "C"
