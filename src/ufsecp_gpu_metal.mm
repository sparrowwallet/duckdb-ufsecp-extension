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
// Generator LUT state (w=16: 16 slices x 65536 entries = 64 MB)
// ============================================================================

static constexpr int GEN_LUT_WBITS = 16;
static constexpr int GEN_LUT_N = (1 << GEN_LUT_WBITS);       // 65536
static constexpr int GEN_LUT_SLICES = (256 + GEN_LUT_WBITS - 1) / GEN_LUT_WBITS; // 16
static constexpr int GEN_LUT_TOTAL = GEN_LUT_SLICES * GEN_LUT_N;
static constexpr int AFFINE_POINT_SIZE = 64;
static constexpr int FIELD_ELEMENT_SIZE = 32;

static mtl::MetalBuffer g_gen_lut_buf;
static bool g_lut_built = false;
static bool g_lut_available = false;
static std::mutex g_lut_mutex;

// LUT build pipelines
static mtl::ComputePipeline g_lut_base_pipeline;
static mtl::ComputePipeline g_lut_build_pipeline;
static mtl::ComputePipeline g_lut_convert_pipeline;

// LUT fused kernel pipeline
static mtl::ComputePipeline g_fused_lut_pipeline;

// Full pipeline state (phases 1-6)
static mtl::MetalBuffer g_spend_buf;
static bool g_spend_uploaded = false;
static mtl::ComputePipeline g_full_pass1_pipeline;
static mtl::ComputePipeline g_batch_inv_pipeline;

// ============================================================================
// Per-batch state (allocated in LaunchBatch, freed in FreeBatch)
// ============================================================================

struct UfsecpMetalBatchState {
    // Fused path: raw LE byte buffers
    mtl::MetalBuffer tweak_buf;
    mtl::MetalBuffer scan_plan_buf;
    mtl::MetalBuffer out_x_buf;
    mtl::MetalBuffer out_y_buf;
    mtl::MetalBuffer midstate_buf;
    mtl::MetalBuffer count_buf;

    // Full pipeline fields (phases 1-6)
    mtl::MetalBuffer output_prefixes_buf;
    mtl::MetalBuffer output_offsets_buf;
    mtl::MetalBuffer output_lengths_buf;
    mtl::MetalBuffer match_flags_buf;
    mtl::MetalBuffer cand_x_buf;
    mtl::MetalBuffer cand_z_buf;
    mtl::MetalBuffer output_pts_buf;
    mtl::MetalBuffer scratch_buf;

    // Multi-dispatch fallback (existing fields)
    std::vector<mtl::HostScalar> scan_scalars;
    std::vector<mtl::HostAffinePoint> tweak_points;

    uint32_t count;
    bool use_fused;
    bool full_pipeline = false;
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
// Lazy LUT construction (called once on first kernel dispatch)
// ============================================================================

static void EnsureGenLutBuilt() {
    if (g_lut_built) return;
    std::lock_guard<std::mutex> lock(g_lut_mutex);
    if (g_lut_built) return;

    if (!g_lut_base_pipeline.valid() || !g_lut_build_pipeline.valid() ||
        !g_lut_convert_pipeline.valid() || !g_fused_lut_pipeline.valid()) {
        g_lut_built = true;
        return;
    }

    // Step 1: Compute GEN_LUT_SLICES base points
    auto bases_buf = g_runtime->alloc_buffer(GEN_LUT_SLICES * AFFINE_POINT_SIZE);
    {
        std::lock_guard<std::mutex> mlock(g_metal_mutex);
        std::vector<mtl::MetalBuffer *> bufs = {&bases_buf};
        g_runtime->dispatch_sync(g_lut_base_pipeline, 1, 1, bufs);
    }

    // Step 2: Allocate LUT (64 MB) + temp H buffer (32 MB)
    auto lut_buf = g_runtime->alloc_buffer(
        (size_t)GEN_LUT_TOTAL * AFFINE_POINT_SIZE);
    auto h_buf = g_runtime->alloc_buffer(
        (size_t)GEN_LUT_TOTAL * FIELD_ELEMENT_SIZE);
    auto n_buf = g_runtime->alloc_buffer(sizeof(int));

    if (!lut_buf.valid() || !h_buf.valid()) {
        g_lut_built = true;
        return;
    }

    int n_entries = GEN_LUT_N;
    n_buf.write(&n_entries, 1);

    // Step 3: Build chain + serial inversion (GEN_LUT_SLICES threadgroups x 1 thread)
    {
        std::lock_guard<std::mutex> mlock(g_metal_mutex);
        std::vector<mtl::MetalBuffer *> bufs = {
            &bases_buf, &lut_buf, &h_buf, &n_buf};
        g_runtime->dispatch_sync(
            g_lut_build_pipeline, GEN_LUT_SLICES, 1, bufs);
    }

    // Step 4: Parallel affine conversion
    int conv_total = GEN_LUT_SLICES * (GEN_LUT_N - 2);
    uint32_t tg = g_lut_convert_pipeline.threadExecutionWidth();
    if (tg == 0) tg = 256;
    {
        std::lock_guard<std::mutex> mlock(g_metal_mutex);
        std::vector<mtl::MetalBuffer *> bufs = {&lut_buf, &h_buf, &n_buf};
        g_runtime->dispatch_sync(
            g_lut_convert_pipeline, conv_total, tg, bufs);
    }

    // Publish
    g_gen_lut_buf = std::move(lut_buf);
    g_lut_available = true;
    g_lut_built = true;

    fprintf(stderr, "[Metal] Generator LUT built (%d MB)\n",
            (int)((size_t)GEN_LUT_TOTAL * AFFINE_POINT_SIZE / (1024 * 1024)));
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
                // Compute BIP352 midstate for GPU kernels
                compute_bip352_midstate(g_bip352_midstate);

                g_use_fused = g_fused_pipeline.valid();
                if (g_use_fused) {
                    fprintf(stderr, "[Metal] Fused BIP-352 kernel available\n");
                } else {
                    fprintf(stderr,
                        "[Metal] Fused kernel unavailable, using multi-dispatch fallback\n");
                }

                // LUT build pipelines (availability checked lazily)
                g_lut_base_pipeline = g_runtime->make_pipeline("compute_lut_base_points");
                g_lut_build_pipeline = g_runtime->make_pipeline("gen_lut_build_affine");
                g_lut_convert_pipeline = g_runtime->make_pipeline("gen_lut_convert_zinv");
                g_fused_lut_pipeline = g_runtime->make_pipeline("bip352_fused_kernel_lut");

                // Full pipeline kernels (phases 1-6)
                g_full_pass1_pipeline = g_runtime->make_pipeline("bip352_full_pass1");
                g_batch_inv_pipeline = g_runtime->make_pipeline("bip352_batch_inv_match");
                if (g_full_pass1_pipeline.valid() && g_batch_inv_pipeline.valid())
                    fprintf(stderr, "[Metal] Full pipeline kernels available\n");
            }
        }
        g_metal_initialized = true;
    }
    *num_gpus = g_metal_device_count;
    return 0;
}

void *UfsecpMetalLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id,
                             const void *precomp) {
    (void)device_id; // Metal runtime manages device selection

    if (!g_runtime || g_metal_device_count == 0)
        return nullptr;

    auto *state = new UfsecpMetalBatchState();
    state->count = count;
    state->use_fused = g_use_fused;

    if (state->use_fused) {
        // Fused path: copy raw LE bytes directly into Metal buffers (no conversion)
        state->tweak_buf = g_runtime->alloc_buffer(count * 64);
        state->scan_plan_buf = g_runtime->alloc_buffer(264);
        state->out_x_buf = g_runtime->alloc_buffer(count * 32);
        state->out_y_buf = g_runtime->alloc_buffer(count * 32);
        state->midstate_buf = g_runtime->alloc_buffer(8 * sizeof(uint32_t));
        state->count_buf = g_runtime->alloc_buffer(sizeof(uint32_t));

        std::memcpy(state->tweak_buf.contents(), tweak_data, count * 64);
        std::memcpy(state->scan_plan_buf.contents(), precomp, 264);
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
        // Build LUT on first use (blocks until complete, thread-safe)
        EnsureGenLutBuilt();

        uint32_t tg = g_fused_pipeline.threadExecutionWidth();
        if (tg == 0) tg = 256;

        if (g_lut_available) {
            std::lock_guard<std::mutex> lock(g_metal_mutex);
            std::vector<mtl::MetalBuffer *> bufs = {
                &state->tweak_buf, &state->scan_plan_buf,
                &state->out_x_buf, &state->out_y_buf,
                &state->midstate_buf, &state->count_buf,
                &g_gen_lut_buf
            };
            g_runtime->dispatch_sync(
                g_fused_lut_pipeline, state->count, tg, bufs);
        } else {
            std::lock_guard<std::mutex> lock(g_metal_mutex);
            std::vector<mtl::MetalBuffer *> bufs = {
                &state->tweak_buf, &state->scan_plan_buf,
                &state->out_x_buf, &state->out_y_buf,
                &state->midstate_buf, &state->count_buf
            };
            g_runtime->dispatch_sync(
                g_fused_pipeline, state->count, tg, bufs);
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

// ============================================================================
// Full pipeline: phases 1-6 on GPU (spend key + batch inversion + matching)
// ============================================================================

void UfsecpMetalSetSpendKey(const uint8_t *spend_xy, int num_labels,
                            const uint8_t *label_keys_xy, int device_id) {
    (void)device_id;
    if (!g_runtime) return;

    // BIP352SpendKeys: base(64) + labels[16](1024) + num_labels(1) + pad(3) = 1092
    static constexpr size_t SPEND_SIZE = 64 + 16 * 64 + 4;
    uint8_t spend_data[SPEND_SIZE] = {};

    std::memcpy(spend_data, spend_xy, 64);

    int n = (num_labels > 16) ? 16 : num_labels;
    if (label_keys_xy && n > 0)
        std::memcpy(spend_data + 64, label_keys_xy, n * 64);

    spend_data[64 + 16 * 64] = (uint8_t)n;

    g_spend_buf = g_runtime->alloc_buffer(SPEND_SIZE);
    std::memcpy(g_spend_buf.contents(), spend_data, SPEND_SIZE);
    g_spend_uploaded = true;
}

void *UfsecpMetalLaunchBatchFull(
    const uint8_t *tweak_data, const int64_t *output_prefixes, uint32_t total_outputs,
    const uint32_t *output_offsets, const uint8_t *output_lengths,
    uint32_t count, int device_id, const void *precomp) {
    (void)device_id;

    if (!g_runtime || g_metal_device_count == 0 ||
        !g_full_pass1_pipeline.valid() || !g_batch_inv_pipeline.valid() ||
        !g_spend_uploaded)
        return nullptr;

    auto *state = new UfsecpMetalBatchState();
    state->count = count;
    state->full_pipeline = true;
    state->use_fused = false;

    // Tweak data + scan plan
    state->tweak_buf = g_runtime->alloc_buffer((size_t)count * 64);
    state->scan_plan_buf = g_runtime->alloc_buffer(264);
    std::memcpy(state->tweak_buf.contents(), tweak_data, (size_t)count * 64);
    std::memcpy(state->scan_plan_buf.contents(), precomp, 264);

    // Output prefix data
    state->output_prefixes_buf = g_runtime->alloc_buffer(
        (size_t)total_outputs * sizeof(int64_t));
    state->output_offsets_buf = g_runtime->alloc_buffer(
        (size_t)count * sizeof(uint32_t));
    state->output_lengths_buf = g_runtime->alloc_buffer((size_t)count);
    std::memcpy(state->output_prefixes_buf.contents(), output_prefixes,
                (size_t)total_outputs * sizeof(int64_t));
    std::memcpy(state->output_offsets_buf.contents(), output_offsets,
                (size_t)count * sizeof(uint32_t));
    std::memcpy(state->output_lengths_buf.contents(), output_lengths,
                (size_t)count);

    // Intermediate buffers
    state->cand_x_buf = g_runtime->alloc_buffer((size_t)count * FIELD_ELEMENT_SIZE);
    state->cand_z_buf = g_runtime->alloc_buffer((size_t)count * FIELD_ELEMENT_SIZE);

    // JacobianPoint in Metal: 3*FieldElement(32) + uint(4) = 100 bytes
    static constexpr size_t JACOBIAN_POINT_SIZE = 3 * 32 + 4;
    state->output_pts_buf = g_runtime->alloc_buffer(
        (size_t)count * JACOBIAN_POINT_SIZE);

    // Match flags
    state->match_flags_buf = g_runtime->alloc_buffer((size_t)count);

    // Midstate + count
    state->midstate_buf = g_runtime->alloc_buffer(8 * sizeof(uint32_t));
    state->count_buf = g_runtime->alloc_buffer(sizeof(uint32_t));
    std::memcpy(state->midstate_buf.contents(), g_bip352_midstate, 32);
    state->count_buf.write(&count, 1);

    // Scratch buffer for batch inversion (device memory instead of threadgroup)
    // 2 * tgsize * FIELD_ELEMENT_SIZE per threadgroup
    uint32_t tgsize = 256;
    uint32_t num_threadgroups = (count + tgsize - 1) / tgsize;
    size_t scratch_size = (size_t)num_threadgroups * 2 * tgsize * FIELD_ELEMENT_SIZE;
    state->scratch_buf = g_runtime->alloc_buffer(scratch_size);

    return state;
}

int UfsecpMetalRunKernelsFull(void *state_handle, uint8_t *match_flags, uint32_t count) {
    auto *state = static_cast<UfsecpMetalBatchState *>(state_handle);
    if (!state || !g_runtime) return -1;

    EnsureGenLutBuilt();
    if (!g_lut_available) return -1;

    // Pass 1: phases 1-5
    {
        std::lock_guard<std::mutex> lock(g_metal_mutex);
        uint32_t tg = g_full_pass1_pipeline.threadExecutionWidth();
        if (tg == 0) tg = 128;
        std::vector<mtl::MetalBuffer *> bufs = {
            &state->tweak_buf, &state->scan_plan_buf,
            &g_gen_lut_buf, &g_spend_buf,
            &state->midstate_buf,
            &state->cand_x_buf, &state->cand_z_buf,
            &state->output_pts_buf, &state->count_buf
        };
        g_runtime->dispatch_sync(g_full_pass1_pipeline, state->count, tg, bufs);
    }

    // Pass 2: batch inversion + matching
    {
        std::lock_guard<std::mutex> lock(g_metal_mutex);
        uint32_t tg = 256;
        std::vector<mtl::MetalBuffer *> bufs = {
            &state->cand_x_buf, &state->cand_z_buf,
            &state->output_pts_buf, &g_spend_buf,
            &state->output_prefixes_buf, &state->output_offsets_buf,
            &state->output_lengths_buf, &state->match_flags_buf,
            &state->count_buf, &state->scratch_buf
        };
        g_runtime->dispatch_sync(g_batch_inv_pipeline, state->count, tg, bufs);
    }

    // Read match flags (unified memory — direct access)
    std::memcpy(match_flags, state->match_flags_buf.contents(), state->count);
    return 0;
}

} // extern "C"
