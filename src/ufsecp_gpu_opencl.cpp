// ============================================================================
// ufsecp_gpu_opencl.cpp — BIP-352 GPU pipeline via UltrafastSecp256k1 OpenCL
// ============================================================================
// Implements the same extern "C" interface as ufsecp_gpu.cu so that
// ProcessBatchGpu in ufsecp_extension.cpp works identically for both backends.
//
// Pipeline (matching CUDA phases 1-4):
//   Phase 1: batch_scalar_mul       — shared_secret = scan_key × tweak[i]  (GPU)
//   Phase 2: batch_jacobian_to_affine                                       (GPU)
//   Phase 3: serialize + tagged SHA-256("BIP0352/SharedSecret", ...)        (CPU)
//   Phase 4: batch_scalar_mul_generator — output = hash × G                (GPU)
//            batch_jacobian_to_affine                                       (GPU)
//
// Phases 5-6 (batch affine add + match) run on CPU in ufsecp_extension.cpp.
// ============================================================================

#include "secp256k1_opencl.hpp"

// UltrafastSecp256k1 CPU headers for tagged hash
#include <secp256k1/tagged_hash.hpp>
#include <secp256k1/sha256.hpp>

#include <mutex>
#include <vector>
#include <cstdint>
#include <cstring>

namespace ocl = secp256k1::opencl;

// ============================================================================
// Global OpenCL context (created once, shared across all batches)
// ============================================================================

static std::unique_ptr<ocl::Context> g_ocl_ctx;
static std::mutex g_ocl_mutex;
static bool g_ocl_initialized = false;
static int g_ocl_device_count = 0;

// BIP0352/SharedSecret tag midstate (computed once)
static secp256k1::SHA256 g_tag_midstate;
static bool g_tag_computed = false;

// ============================================================================
// Per-batch state (allocated in LaunchBatch, freed in FreeBatch)
// ============================================================================

struct UfsecpOclBatchState {
	std::vector<ocl::Scalar> scan_scalars;      // N copies of scan_scalar
	std::vector<ocl::AffinePoint> tweak_points;  // N affine points from input
	uint32_t count;
};

// ============================================================================
// Byte-order conversion helpers
// ============================================================================

// LE bytes (Frigate wire format) → OpenCL Scalar (LE limbs)
// Wire: bytes[0] = LSB. Limb[i] = bytes[i*8..i*8+7] as uint64.
static ocl::Scalar scalar_from_le(const uint8_t *le32) {
	ocl::Scalar s;
	for (int i = 0; i < 4; i++) {
		uint64_t v = 0;
		for (int j = 0; j < 8; j++)
			v |= (uint64_t)le32[i * 8 + j] << (j * 8);
		s.limbs[i] = v;
	}
	return s;
}

// BE bytes (SHA-256 output) → OpenCL Scalar (LE limbs)
static ocl::Scalar scalar_from_be(const uint8_t *be32) {
	ocl::Scalar s;
	for (int i = 0; i < 4; i++) {
		uint64_t v = 0;
		for (int j = 0; j < 8; j++)
			v |= (uint64_t)be32[31 - (i * 8 + j)] << (j * 8);
		s.limbs[i] = v;
	}
	return s;
}

// LE bytes pair (32+32) → OpenCL AffinePoint (LE limbs)
static ocl::AffinePoint affine_from_le(const uint8_t *xy64) {
	ocl::AffinePoint ap;
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

// OpenCL AffinePoint → 33-byte compressed SEC1 (for tagged hash serialization)
static void affine_to_compressed(const ocl::AffinePoint &ap, uint8_t *out33) {
	// x: LE limbs → BE bytes
	for (int i = 0; i < 4; i++) {
		uint64_t v = ap.x.limbs[i];
		for (int j = 0; j < 8; j++)
			out33[32 - (i * 8 + j)] = (uint8_t)(v >> (j * 8));
	}
	// prefix from y parity
	out33[0] = (ap.y.limbs[0] & 1) ? 0x03 : 0x02;
}

// OpenCL AffinePoint → LE bytes (for CPU consumption in ProcessBatchGpu)
static void affine_to_le(const ocl::AffinePoint &ap, uint8_t *out_x, uint8_t *out_y) {
	for (int i = 0; i < 4; i++) {
		uint64_t xv = ap.x.limbs[i], yv = ap.y.limbs[i];
		for (int j = 0; j < 8; j++) {
			out_x[i * 8 + j] = (uint8_t)(xv >> (j * 8));
			out_y[i * 8 + j] = (uint8_t)(yv >> (j * 8));
		}
	}
}

// ============================================================================
// Extern "C" interface — same signatures as ufsecp_gpu.cu
// ============================================================================

extern "C" {

int UfsecpOclDetect(int *num_gpus) {
	std::lock_guard<std::mutex> lock(g_ocl_mutex);
	if (!g_ocl_initialized) {
		ocl::DeviceConfig config;
		config.verbose = false;
		config.max_batch_size = 1000000;
		g_ocl_ctx = ocl::Context::create(config);
		if (g_ocl_ctx && g_ocl_ctx->is_valid()) {
			g_ocl_device_count = 1;
		}
		g_ocl_initialized = true;
	}
	*num_gpus = g_ocl_device_count;
	return 0;
}

void *UfsecpOclLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id) {
	(void)device_id; // OpenCL context manages device selection

	if (!g_ocl_ctx || !g_ocl_ctx->is_valid())
		return nullptr;

	auto *state = new UfsecpOclBatchState();
	state->count = count;

	// Convert scan key (LE bytes → OpenCL Scalar), replicate for batch API
	ocl::Scalar scan_scalar = scalar_from_le(scan_key);
	state->scan_scalars.resize(count, scan_scalar);

	// Convert tweak points (N × 64 LE bytes → OpenCL AffinePoints)
	state->tweak_points.resize(count);
	for (uint32_t i = 0; i < count; i++)
		state->tweak_points[i] = affine_from_le(tweak_data + i * 64);

	return state;
}

int UfsecpOclRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count) {
	auto *state = static_cast<UfsecpOclBatchState *>(state_handle);
	if (!state || !g_ocl_ctx)
		return -1;

	// Compute tag midstate once (thread-safe: worst case is redundant computation)
	if (!g_tag_computed) {
		g_tag_midstate = secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");
		g_tag_computed = true;
	}

	// Phase 1: batch scalar multiply — shared[i] = scan_scalar × tweak[i]
	std::vector<ocl::JacobianPoint> jac1(count);
	g_ocl_ctx->batch_scalar_mul(state->scan_scalars.data(), state->tweak_points.data(), jac1.data(), count);

	// Phase 2: Jacobian → affine (for SEC1 serialization)
	std::vector<ocl::AffinePoint> aff1(count);
	g_ocl_ctx->batch_jacobian_to_affine(jac1.data(), aff1.data(), count);

	// Phase 3: CPU — serialize compressed + tagged SHA-256 → hash scalars
	std::vector<ocl::Scalar> hash_scalars(count);
	for (uint32_t i = 0; i < count; i++) {
		uint8_t ser[37];
		affine_to_compressed(aff1[i], ser);
		std::memset(ser + 33, 0, 4); // output index k=0

		auto hash = secp256k1::detail::cached_tagged_hash(g_tag_midstate, ser, 37);
		hash_scalars[i] = scalar_from_be(hash.data());
	}

	// Phase 4: batch generator multiply — output[i] = hash[i] × G
	std::vector<ocl::JacobianPoint> jac2(count);
	g_ocl_ctx->batch_scalar_mul_generator(hash_scalars.data(), jac2.data(), count);

	// Phase 4b: Jacobian → affine (output for CPU phases 5-6)
	std::vector<ocl::AffinePoint> aff2(count);
	g_ocl_ctx->batch_jacobian_to_affine(jac2.data(), aff2.data(), count);

	// Convert OpenCL affine points to LE bytes for CPU consumption
	for (uint32_t i = 0; i < count; i++)
		affine_to_le(aff2[i], out_x + i * 32, out_y + i * 32);

	return 0;
}

void UfsecpOclFreeBatch(void *state_handle) {
	if (!state_handle)
		return;
	delete static_cast<UfsecpOclBatchState *>(state_handle);
}

} // extern "C"
