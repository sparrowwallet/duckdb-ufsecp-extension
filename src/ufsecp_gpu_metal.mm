// ============================================================================
// ufsecp_gpu_metal.mm -- BIP-352 GPU pipeline via UltrafastSecp256k1 Metal
// ============================================================================
// Implements the same extern "C" interface as ufsecp_gpu_opencl.cpp so that
// ProcessBatchGpu in ufsecp_extension.cpp works identically for all backends.
//
// Preferred path: fused single-dispatch kernel (bip352_fused_kernel)
//   All 5 phases run on GPU in one dispatch per thread.
//
// Fallback path: multi-dispatch pipeline
//   Phase 1: scalar_mul_batch         -- shared_secret = scan_key * tweak[i]  (GPU)
//   Phase 2: serialize + tagged SHA-256("BIP0352/SharedSecret", ...)          (CPU)
//   Phase 3: generator_mul_batch      -- output = hash * G                    (GPU)
//
// Phases 5-6 (batch affine add + match) run on CPU in ufsecp_extension.cpp.
// ============================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_runtime.h"
#include "host_helpers.h"

// UltrafastSecp256k1 CPU headers for tagged hash
#include <secp256k1/tagged_hash.hpp>
#include <secp256k1/sha256.hpp>

#include <openssl/sha.h>

#include <mutex>
#include <vector>
#include <cstdint>
#include <cstring>

#include "metal_shader_source.h"

namespace mtl = secp256k1::metal;

// ============================================================================
// Global Metal state (created once, shared across all batches)
// ============================================================================

static std::unique_ptr<mtl::MetalRuntime> g_runtime;
static mtl::ComputePipeline g_scalar_mul_pipeline;
static mtl::ComputePipeline g_generator_mul_pipeline;
static mtl::ComputePipeline g_fused_pipeline;
static bool g_use_fused = false;
static std::mutex g_metal_mutex;
static bool g_metal_initialized = false;
static int g_metal_device_count = 0;

// BIP0352/SharedSecret tag midstate (computed once)
static secp256k1::SHA256 g_tag_midstate;
static bool g_tag_computed = false;

// BIP0352/SharedSecret midstate as 8 x uint32_t for the fused GPU kernel
static uint32_t g_bip352_midstate[8];

// ============================================================================
// Per-batch state (allocated in LaunchBatch, freed in FreeBatch)
// ============================================================================

struct UfsecpMetalBatchState {
    // Fused path: raw LE byte buffers
    mtl::MetalBuffer tweak_buf;
    mtl::MetalBuffer scan_key_buf;
    mtl::MetalBuffer out_x_buf;
    mtl::MetalBuffer out_y_buf;
    mtl::MetalBuffer midstate_buf;
    mtl::MetalBuffer count_buf;

    // Multi-dispatch fallback (existing fields)
    std::vector<mtl::HostScalar> scan_scalars;
    std::vector<mtl::HostAffinePoint> tweak_points;

    uint32_t count;
    bool use_fused;
};

// ============================================================================
// Byte-order conversion helpers (used by multi-dispatch fallback)
// ============================================================================

// LE bytes (Frigate wire format) -> Metal HostScalar (LE limbs)
// Wire: bytes[0] = LSB. Limb[i] = bytes[i*8..i*8+7] as uint64.
static mtl::HostScalar scalar_from_le(const uint8_t *le32) {
    mtl::HostScalar s;
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++)
            v |= (uint64_t)le32[i * 8 + j] << (j * 8);
        s.limbs[i] = v;
    }
    return s;
}

// BE bytes (SHA-256 output) -> Metal HostScalar (LE limbs)
static mtl::HostScalar scalar_from_be(const uint8_t *be32) {
    mtl::HostScalar s;
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++)
            v |= (uint64_t)be32[31 - (i * 8 + j)] << (j * 8);
        s.limbs[i] = v;
    }
    return s;
}

// LE bytes pair (32+32) -> Metal HostAffinePoint (LE limbs)
static mtl::HostAffinePoint affine_from_le(const uint8_t *xy64) {
    mtl::HostAffinePoint ap;
    for (int i = 0; i < 4; i++) {
        uint64_t xv = 0, yv = 0;
        for (int j = 0; j < 8; j++) {
            xv |= (uint64_t)xy64[i * 8 + j] << (j * 8);
            yv |= (uint64_t)xy64[32 + i * 8 + j] << (j * 8);
        }
        ap.x.limbs[i] = xv;
        ap.y.limbs[i] = yv;
    }
    return ap;
}

// Metal HostAffinePoint -> 33-byte compressed SEC1 (for tagged hash serialization)
static void affine_to_compressed(const mtl::HostAffinePoint &ap, uint8_t *out33) {
    // x: LE limbs -> BE bytes
    for (int i = 0; i < 4; i++) {
        uint64_t v = ap.x.limbs[i];
        for (int j = 0; j < 8; j++)
            out33[32 - (i * 8 + j)] = (uint8_t)(v >> (j * 8));
    }
    // prefix from y parity
    out33[0] = (ap.y.limbs[0] & 1) ? 0x03 : 0x02;
}

// Metal HostAffinePoint -> LE bytes (for CPU consumption in ProcessBatchGpu)
static void affine_to_le(const mtl::HostAffinePoint &ap, uint8_t *out_x, uint8_t *out_y) {
    for (int i = 0; i < 4; i++) {
        uint64_t xv = ap.x.limbs[i], yv = ap.y.limbs[i];
        for (int j = 0; j < 8; j++) {
            out_x[i * 8 + j] = (uint8_t)(xv >> (j * 8));
            out_y[i * 8 + j] = (uint8_t)(yv >> (j * 8));
        }
    }
}

// ============================================================================
// BIP352 midstate computation using OpenSSL SHA-256
// ============================================================================

static void compute_bip352_midstate(uint32_t out[8]) {
    // SHA256("BIP0352/SharedSecret") -> tag_hash
    unsigned char tag_hash[32];
    SHA256(reinterpret_cast<const unsigned char *>("BIP0352/SharedSecret"), 20, tag_hash);

    // Build 64-byte block: tag_hash || tag_hash
    unsigned char block[64];
    std::memcpy(block, tag_hash, 32);
    std::memcpy(block + 32, tag_hash, 32);

    // Process one block through SHA-256 to get the midstate
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, block, 64);
    // ctx.h now contains the midstate (after processing the 64-byte block)
    for (int i = 0; i < 8; i++)
        out[i] = ctx.h[i];
}

// ============================================================================
// Extern "C" interface -- same signatures as ufsecp_gpu_opencl.cpp
// ============================================================================

extern "C" {

int UfsecpMetalDetect(int *num_gpus) {
    std::lock_guard<std::mutex> lock(g_metal_mutex);
    if (!g_metal_initialized) {
        g_runtime = std::make_unique<mtl::MetalRuntime>();
        if (g_runtime->init()) {
            // Compile shaders from embedded source
            std::string source(METAL_SHADER_SOURCE);
            if (g_runtime->load_library_from_source(source)) {
                g_scalar_mul_pipeline = g_runtime->make_pipeline("scalar_mul_batch");
                g_generator_mul_pipeline = g_runtime->make_pipeline("generator_mul_batch");
                if (g_scalar_mul_pipeline.valid() && g_generator_mul_pipeline.valid()) {
                    g_metal_device_count = 1;
                    auto info = g_runtime->device_info();
                    fprintf(stderr, "[Metal] GPU detected: %s\n", info.name.c_str());
                }

                // Try to create the fused pipeline
                g_fused_pipeline = g_runtime->make_pipeline("bip352_fused_kernel");
                g_use_fused = g_fused_pipeline.valid();
                if (g_use_fused) {
                    fprintf(stderr, "[Metal] Fused BIP-352 kernel available\n");
                    compute_bip352_midstate(g_bip352_midstate);
                } else {
                    fprintf(stderr, "[Metal] Fused kernel unavailable, using multi-dispatch fallback\n");
                }
            }
        }
        g_metal_initialized = true;
    }
    *num_gpus = g_metal_device_count;
    return 0;
}

void *UfsecpMetalLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id) {
    (void)device_id; // Metal runtime manages device selection

    if (!g_runtime || g_metal_device_count == 0)
        return nullptr;

    auto *state = new UfsecpMetalBatchState();
    state->count = count;
    state->use_fused = g_use_fused;

    if (state->use_fused) {
        // Fused path: copy raw LE bytes directly into Metal buffers (no conversion)
        state->tweak_buf = g_runtime->alloc_buffer(count * 64);
        state->scan_key_buf = g_runtime->alloc_buffer(32);
        state->out_x_buf = g_runtime->alloc_buffer(count * 32);
        state->out_y_buf = g_runtime->alloc_buffer(count * 32);
        state->midstate_buf = g_runtime->alloc_buffer(8 * sizeof(uint32_t));
        state->count_buf = g_runtime->alloc_buffer(sizeof(uint32_t));

        std::memcpy(state->tweak_buf.contents(), tweak_data, count * 64);
        std::memcpy(state->scan_key_buf.contents(), scan_key, 32);
        std::memcpy(state->midstate_buf.contents(), g_bip352_midstate, 8 * sizeof(uint32_t));
        state->count_buf.write(&count, 1);
    } else {
        // Multi-dispatch fallback: convert to host types
        mtl::HostScalar scan_scalar = scalar_from_le(scan_key);
        state->scan_scalars.resize(count, scan_scalar);

        state->tweak_points.resize(count);
        for (uint32_t i = 0; i < count; i++)
            state->tweak_points[i] = affine_from_le(tweak_data + i * 64);
    }

    return state;
}

int UfsecpMetalRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count) {
    auto *state = static_cast<UfsecpMetalBatchState *>(state_handle);
    if (!state || !g_runtime)
        return -1;

    // ====================================================================
    // Fused path: single GPU dispatch
    // ====================================================================
    if (state->use_fused) {
        uint32_t tg = g_fused_pipeline.threadExecutionWidth();
        if (tg == 0) tg = 256;

        {
            std::lock_guard<std::mutex> lock(g_metal_mutex);
            std::vector<mtl::MetalBuffer *> bufs = {
                &state->tweak_buf, &state->scan_key_buf,
                &state->out_x_buf, &state->out_y_buf,
                &state->midstate_buf, &state->count_buf
            };
            g_runtime->dispatch_sync(g_fused_pipeline, state->count, tg, bufs);
        }

        std::memcpy(out_x, state->out_x_buf.contents(), state->count * 32);
        std::memcpy(out_y, state->out_y_buf.contents(), state->count * 32);
        return 0;
    }

    // ====================================================================
    // Multi-dispatch fallback path
    // ====================================================================

    // Compute tag midstate once (thread-safe: worst case is redundant computation)
    if (!g_tag_computed) {
        g_tag_midstate = secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");
        g_tag_computed = true;
    }

    const int affine_size = 64;  // AffinePoint: 2 x FieldElement = 64 bytes
    const int scalar_size = 32;  // Scalar256: 8 x uint32_t = 32 bytes

    uint32_t tg = g_scalar_mul_pipeline.threadExecutionWidth();
    if (tg == 0) tg = 256;

    // Phase 1: GPU scalar_mul_batch -- shared[i] = scan_scalar * tweak[i]
    auto bases_buf = g_runtime->alloc_buffer(count * affine_size);
    auto scalars_buf = g_runtime->alloc_buffer(count * scalar_size);
    auto results_buf = g_runtime->alloc_buffer(count * affine_size);
    auto cnt_buf = g_runtime->alloc_buffer(4);

    std::memcpy(bases_buf.contents(), state->tweak_points.data(), count * affine_size);
    std::memcpy(scalars_buf.contents(), state->scan_scalars.data(), count * scalar_size);
    cnt_buf.write(&count, 1);

    {
        std::lock_guard<std::mutex> lock(g_metal_mutex);
        std::vector<mtl::MetalBuffer *> smul_bufs = {&bases_buf, &scalars_buf, &results_buf, &cnt_buf};
        g_runtime->dispatch_sync(g_scalar_mul_pipeline, count, tg, smul_bufs);
    }

    // Phase 2: Read affine results (unified memory -- direct access)
    auto *shared_data = static_cast<const uint8_t *>(results_buf.contents());

    // Phase 3: CPU -- serialize compressed + tagged SHA-256 -> hash scalars
    auto hash_scalars_buf = g_runtime->alloc_buffer(count * scalar_size);
    auto *hash_data = static_cast<uint8_t *>(hash_scalars_buf.contents());

    for (uint32_t i = 0; i < count; i++) {
        const auto *ap = reinterpret_cast<const mtl::HostAffinePoint *>(shared_data + i * affine_size);
        // Serialize to compressed SEC1 + 4 zero bytes (output index k=0)
        uint8_t ser[37];
        affine_to_compressed(*ap, ser);
        std::memset(ser + 33, 0, 4);

        auto hash = secp256k1::detail::cached_tagged_hash(g_tag_midstate, ser, 37);
        auto hs = scalar_from_be(hash.data());
        std::memcpy(hash_data + i * scalar_size, &hs, scalar_size);
    }

    // Phase 4: GPU generator_mul_batch -- output[i] = hash[i] * G
    auto gen_results_buf = g_runtime->alloc_buffer(count * affine_size);
    auto gen_cnt_buf = g_runtime->alloc_buffer(4);
    gen_cnt_buf.write(&count, 1);

    uint32_t tg_gen = g_generator_mul_pipeline.threadExecutionWidth();
    if (tg_gen == 0) tg_gen = 256;

    {
        std::lock_guard<std::mutex> lock(g_metal_mutex);
        std::vector<mtl::MetalBuffer *> gmul_bufs = {&hash_scalars_buf, &gen_results_buf, &gen_cnt_buf};
        g_runtime->dispatch_sync(g_generator_mul_pipeline, count, tg_gen, gmul_bufs);
    }

    // Convert Metal affine points to LE bytes for CPU consumption
    auto *out_data = static_cast<const uint8_t *>(gen_results_buf.contents());
    for (uint32_t i = 0; i < count; i++) {
        const auto *ap = reinterpret_cast<const mtl::HostAffinePoint *>(out_data + i * affine_size);
        affine_to_le(*ap, out_x + i * 32, out_y + i * 32);
    }

    return 0;
}

void UfsecpMetalFreeBatch(void *state_handle) {
    if (!state_handle)
        return;
    delete static_cast<UfsecpMetalBatchState *>(state_handle);
}

} // extern "C"
