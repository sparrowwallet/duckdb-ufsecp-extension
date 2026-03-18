// ============================================================================
// ufsecp_gpu_opencl.cpp — BIP-352 GPU pipeline via UltrafastSecp256k1 OpenCL
// ============================================================================
// Implements the same extern "C" interface as ufsecp_gpu.cu so that
// ProcessBatchGpu in ufsecp_extension.cpp works identically for both backends.
//
// Preferred path: fused single-dispatch kernel (bip352_fused_kernel)
//   All 5 phases run on GPU in one dispatch per thread.
//
// Fallback path: multi-dispatch pipeline
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

// Raw OpenCL API for fused kernel compilation
#include <CL/cl.h>

#include <mutex>
#include <vector>
#include <cstdint>
#include <cstring>

#include "opencl_fused_kernel_source.h"

namespace ocl = secp256k1::opencl;

// ============================================================================
// Global OpenCL context (created once, shared across all batches)
// ============================================================================

static std::unique_ptr<ocl::Context> g_ocl_ctx;
static std::mutex g_ocl_mutex;
static bool g_ocl_initialized = false;
static int g_ocl_device_count = 0;

// BIP0352/SharedSecret tag midstate (computed once, for multi-dispatch fallback)
static secp256k1::SHA256 g_tag_midstate;
static bool g_tag_computed = false;

// Fused kernel state
static cl_program g_fused_program = nullptr;
static cl_kernel g_fused_kernel = nullptr;
static bool g_use_fused = false;

// BIP0352/SharedSecret midstate as 8 x uint32_t for the fused GPU kernel
static const uint32_t g_bip352_midstate[8] = {0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
                                              0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU};

// ============================================================================
// Per-batch state (allocated in LaunchBatch, freed in FreeBatch)
// ============================================================================

struct UfsecpOclBatchState {
	// Fused path: raw OpenCL buffers
	cl_mem tweak_buf;
	cl_mem scan_key_buf;
	cl_mem out_x_buf;
	cl_mem out_y_buf;
	cl_mem midstate_buf;

	// Multi-dispatch fallback (existing fields)
	std::vector<ocl::Scalar> scan_scalars;
	std::vector<ocl::AffinePoint> tweak_points;

	uint32_t count;
	bool use_fused;
};

// ============================================================================
// Byte-order conversion helpers (used by multi-dispatch fallback)
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

			// Try to compile the fused kernel using the library's OpenCL context
			auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());
			auto *queue = static_cast<cl_command_queue>(g_ocl_ctx->native_queue());
			if (ctx && queue) {
				// Get device from command queue
				cl_device_id device = nullptr;
				clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr);

				if (device) {
					const char *src = OPENCL_FUSED_KERNEL_SOURCE;
					size_t src_len = strlen(src);
					cl_int err;
					g_fused_program = clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
					if (err == CL_SUCCESS) {
						err = clBuildProgram(g_fused_program, 1, &device,
						                     "-cl-std=CL1.2 -cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
						if (err == CL_SUCCESS) {
							g_fused_kernel = clCreateKernel(g_fused_program, "bip352_fused_kernel", &err);
							if (err == CL_SUCCESS) {
								g_use_fused = true;
								fprintf(stderr, "[OpenCL] Fused BIP-352 kernel available\n");
							}
						} else {
							// Log build error for debugging
							size_t log_size = 0;
							clGetProgramBuildInfo(g_fused_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
							if (log_size > 1) {
								std::vector<char> log(log_size);
								clGetProgramBuildInfo(g_fused_program, device, CL_PROGRAM_BUILD_LOG, log_size,
								                      log.data(), nullptr);
								fprintf(stderr, "[OpenCL] Fused kernel build failed:\n%s\n", log.data());
							}
						}
					}
					if (!g_use_fused) {
						fprintf(stderr, "[OpenCL] Fused kernel unavailable, using multi-dispatch fallback\n");
						if (g_fused_program) {
							clReleaseProgram(g_fused_program);
							g_fused_program = nullptr;
						}
					}
				}
			}
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
	state->use_fused = g_use_fused;

	if (state->use_fused) {
		// Fused path: copy raw LE bytes directly into OpenCL buffers (no conversion)
		auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());
		cl_int err;

		state->tweak_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * 64,
		                                  const_cast<uint8_t *>(tweak_data), &err);
		if (err != CL_SUCCESS) {
			delete state;
			return nullptr;
		}

		state->scan_key_buf =
		    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, const_cast<uint8_t *>(scan_key), &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf);
			delete state;
			return nullptr;
		}

		state->out_x_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, count * 32, nullptr, &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf);
			clReleaseMemObject(state->scan_key_buf);
			delete state;
			return nullptr;
		}

		state->out_y_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, count * 32, nullptr, &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf);
			clReleaseMemObject(state->scan_key_buf);
			clReleaseMemObject(state->out_x_buf);
			delete state;
			return nullptr;
		}

		state->midstate_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8 * sizeof(uint32_t),
		                                     const_cast<uint32_t *>(g_bip352_midstate), &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf);
			clReleaseMemObject(state->scan_key_buf);
			clReleaseMemObject(state->out_x_buf);
			clReleaseMemObject(state->out_y_buf);
			delete state;
			return nullptr;
		}
	} else {
		// Multi-dispatch fallback: convert to library types
		ocl::Scalar scan_scalar = scalar_from_le(scan_key);
		state->scan_scalars.resize(count, scan_scalar);

		state->tweak_points.resize(count);
		for (uint32_t i = 0; i < count; i++)
			state->tweak_points[i] = affine_from_le(tweak_data + i * 64);

		// Zero out buffer handles so FreeBatch doesn't release them
		state->tweak_buf = nullptr;
		state->scan_key_buf = nullptr;
		state->out_x_buf = nullptr;
		state->out_y_buf = nullptr;
		state->midstate_buf = nullptr;
	}

	return state;
}

int UfsecpOclRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count) {
	auto *state = static_cast<UfsecpOclBatchState *>(state_handle);
	if (!state || !g_ocl_ctx)
		return -1;

	// ====================================================================
	// Fused path: single GPU dispatch
	// ====================================================================
	if (state->use_fused) {
		auto *queue = static_cast<cl_command_queue>(g_ocl_ctx->native_queue());

		// Lock around setargs + enqueue to prevent concurrent threads from
		// interleaving kernel arguments on the shared cl_kernel object.
		std::lock_guard<std::mutex> lock(g_ocl_mutex);

		// Set kernel arguments
		clSetKernelArg(g_fused_kernel, 0, sizeof(cl_mem), &state->tweak_buf);
		clSetKernelArg(g_fused_kernel, 1, sizeof(cl_mem), &state->scan_key_buf);
		clSetKernelArg(g_fused_kernel, 2, sizeof(cl_mem), &state->out_x_buf);
		clSetKernelArg(g_fused_kernel, 3, sizeof(cl_mem), &state->out_y_buf);
		clSetKernelArg(g_fused_kernel, 4, sizeof(cl_mem), &state->midstate_buf);
		clSetKernelArg(g_fused_kernel, 5, sizeof(uint32_t), &count);

		// Query preferred work group size
		size_t local_size = 256;
		clGetKernelWorkGroupInfo(g_fused_kernel, nullptr, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size,
		                         nullptr);
		if (local_size > 256)
			local_size = 256;

		// Round global size up to multiple of local size
		size_t global_size = ((size_t)count + local_size - 1) / local_size * local_size;

		cl_int err =
		    clEnqueueNDRangeKernel(queue, g_fused_kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
		if (err != CL_SUCCESS)
			return -1;

		// Read back results
		clEnqueueReadBuffer(queue, state->out_x_buf, CL_TRUE, 0, count * 32, out_x, 0, nullptr, nullptr);
		clEnqueueReadBuffer(queue, state->out_y_buf, CL_TRUE, 0, count * 32, out_y, 0, nullptr, nullptr);
		clFinish(queue);

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
	auto *state = static_cast<UfsecpOclBatchState *>(state_handle);
	if (state->tweak_buf)
		clReleaseMemObject(state->tweak_buf);
	if (state->scan_key_buf)
		clReleaseMemObject(state->scan_key_buf);
	if (state->out_x_buf)
		clReleaseMemObject(state->out_x_buf);
	if (state->out_y_buf)
		clReleaseMemObject(state->out_y_buf);
	if (state->midstate_buf)
		clReleaseMemObject(state->midstate_buf);
	delete state;
}

} // extern "C"
