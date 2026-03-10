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
//   Phase 4: output_point = hash × G                 (scalar_mul_generator_const)
//
// Phases 5-6 (batch affine add + match) run on CPU.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// UltrafastSecp256k1 CUDA headers
#include "secp256k1.cuh"    // FieldElement, Scalar, JacobianPoint, scalar_mul_glv_wnaf,
                            // scalar_mul_generator_const, field_inv/sqr/mul, field_set_one
#include "ecdsa.cuh"        // scalar_from_bytes, field_to_bytes, SHA256Ctx, sha256_*
#include "affine_add.cuh"   // jacobian_to_affine

using namespace secp256k1::cuda;

// ============================================================================
// BIP0352/SharedSecret tagged hash midstate
// ============================================================================
// Precomputed SHA-256 state after processing:
//   SHA256("BIP0352/SharedSecret") || SHA256("BIP0352/SharedSecret")
// which is exactly one 64-byte SHA-256 block.
//
// tag_hash = SHA256("BIP0352/SharedSecret")
//          = 9f6d8011581eb62d72e613604c330dca2a0bd349e24a46d9a2ef24b9a98f41bd
//
// Verified: tagged_hash_midstate(37 zero bytes) produces
//           bdbc9499e46a3bba23adc93d904bef650f28dbf5ae080296cf84c3d1f97e655c
//           matching the full SHA256(tag_hash || tag_hash || msg) computation.

__device__ __constant__ static uint32_t BIP352_MIDSTATE[8] = {
    0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
    0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU
};

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
// Fused BIP-352 kernel — one thread per transaction
// ============================================================================

__global__ void BIP352FusedKernel(
    const uint8_t* __restrict__ tweak_xy,   // N × 64 bytes
    const uint8_t* __restrict__ scan_key,   // 32 bytes (same for all threads)
    uint8_t* __restrict__ out_x,            // N × 32 bytes
    uint8_t* __restrict__ out_y,            // N × 32 bytes
    uint32_t count)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // ----------------------------------------------------------------
    // Load inputs (LE wire format → internal representation)
    // ----------------------------------------------------------------

    // Load tweak key: LE bytes → FieldElement (4×u64 LE limbs)
    // Wire: bytes[0] = LSB. FieldElement: limbs[0] = least-significant u64.
    // Each u64 limb spans 8 LE bytes: limb[i] = bytes[i*8..i*8+7].
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
    field_set_one(&tweak_point.z);  // affine → Z = 1
    tweak_point.infinity = false;

    // Load scan private key (LE bytes → BE bytes → Scalar via scalar_from_bytes)
    uint8_t sk_be[32];
    for (int i = 0; i < 32; i++)
        sk_be[i] = scan_key[31 - i];
    Scalar sk;
    scalar_from_bytes(sk_be, &sk);

    // ----------------------------------------------------------------
    // Phase 1: Shared secret — ONE scalar multiplication
    // ----------------------------------------------------------------
    // scalar_mul_glv_wnaf: GLV endomorphism + Shamir's interleaved wNAF.
    // ~128 doublings + ~64 mixed additions ≈ 1728 field-mul-eq.
    JacobianPoint shared_point;
    scalar_mul_glv_wnaf(&tweak_point, &sk, &shared_point);

    // ----------------------------------------------------------------
    // Phase 2: Jacobian → affine → SEC1 compressed serialization
    // ----------------------------------------------------------------
    // x = X/Z², y = Y/Z³  (1 field_inv + 1 sqr + 3 mul)
    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv(&shared_point.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&shared_point.x, &z_inv2, &x_aff);
    field_mul(&shared_point.y, &z_inv3, &y_aff);

    // X coordinate → big-endian bytes for SEC1
    uint8_t shared_x[32];
    field_to_bytes(&x_aff, shared_x);

    // Y parity → SEC1 prefix (0x02 = even, 0x03 = odd)
    // field_to_bytes normalizes mod p; byte[31] is LSB → bit 0 is parity.
    uint8_t y_bytes[32];
    field_to_bytes(&y_aff, y_bytes);
    uint8_t prefix = 0x02 + (y_bytes[31] & 1);

    // Build 37-byte buffer: [prefix || x_BE || 0x00000000]
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
    ctx.total = 64;  // 64 bytes already processed by midstate

    sha256_update(&ctx, serialized, 37);
    uint8_t hash[32];
    sha256_final(&ctx, hash);

    // ----------------------------------------------------------------
    // Phase 4: Generator multiplication — hash × G
    // ----------------------------------------------------------------
    // scalar_mul_generator_const: precomputed table in __constant__ memory.
    Scalar hash_scalar;
    scalar_from_bytes(hash, &hash_scalar);

    JacobianPoint output_point;
    scalar_mul_generator_const(&hash_scalar, &output_point);

    // Convert Jacobian → affine (second field inversion per thread)
    jacobian_to_affine(&output_point.x, &output_point.y, &output_point.z);

    // ----------------------------------------------------------------
    // Write output: affine (x, y) as LE bytes for CPU consumption
    // ----------------------------------------------------------------
    // field_to_bytes gives big-endian; convert to LE wire format.
    uint8_t ox_be[32], oy_be[32];
    field_to_bytes(&output_point.x, ox_be);
    field_to_bytes(&output_point.y, oy_be);

    uint8_t* dst_x = out_x + idx * 32;
    uint8_t* dst_y = out_y + idx * 32;
    for (int i = 0; i < 32; i++) {
        dst_x[i] = ox_be[31 - i];  // BE → LE
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

    // Launch fused kernel
    int threads = 256;
    int blocks = ((int)count + threads - 1) / threads;

    BIP352FusedKernel<<<blocks, threads, 0, state->stream>>>(
        state->d_tweak_xy,
        state->d_scan_key,
        state->d_output_x,
        state->d_output_y,
        count
    );

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
