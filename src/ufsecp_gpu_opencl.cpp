// ============================================================================
// ufsecp_gpu_opencl.cpp — BIP-352 GPU pipeline via UltrafastSecp256k1 OpenCL
// ============================================================================
// Implements the same extern "C" interface as ufsecp_gpu.cu so that
// ProcessBatchGpu in ufsecp_extension.cpp works identically for both backends.
//
// Preferred path: LUT fused kernel (bip352_fused_kernel_lut)
//   64 MB precomputed generator table, 15 additions for k*G.
//
// Fallback path 1: GLV fused kernel (bip352_fused_kernel)
//   All 5 phases in one dispatch using GLV generator multiplication.
//
// Fallback path 2: multi-dispatch pipeline
//   4 GPU dispatches + 1 CPU phase (original approach).
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
#include <cstdio>
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

// LUT kernel state
static cl_kernel g_fused_kernel_lut = nullptr;
static cl_kernel g_lut_base_kernel = nullptr;
static cl_kernel g_lut_build_kernel = nullptr;
static cl_kernel g_lut_convert_kernel = nullptr;
static bool g_use_lut = false;

// Generator LUT (built once, persistent)
static cl_mem g_ocl_gen_lut = nullptr;
static bool g_ocl_lut_built = false;
static std::mutex g_ocl_lut_mutex;

// BIP0352/SharedSecret midstate as 8 x uint32_t for the fused GPU kernel
static const uint32_t g_bip352_midstate[8] = {0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
                                              0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU};

// ============================================================================
// Generator LUT construction (lazy, thread-safe)
// ============================================================================

static void EnsureOclGenLutBuilt() {
	if (g_ocl_lut_built) return;
	std::lock_guard<std::mutex> lock(g_ocl_lut_mutex);
	if (g_ocl_lut_built) return;

	auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());
	auto *queue = static_cast<cl_command_queue>(g_ocl_ctx->native_queue());
	cl_int err;

	// AffinePoint = 64 bytes, FieldElement = 32 bytes
	static constexpr int N = 65536;
	static constexpr int SLICES = 16;
	static constexpr int TOTAL = SLICES * N;
	static constexpr size_t AFFINE_SIZE = 64;
	static constexpr size_t FIELD_SIZE = 32;

	// Allocate buffers (buffer creation doesn't touch the command queue)
	cl_mem d_bases = clCreateBuffer(ctx, CL_MEM_READ_WRITE, SLICES * AFFINE_SIZE, nullptr, &err);
	if (err != CL_SUCCESS) { g_ocl_lut_built = true; return; }

	cl_mem d_lut = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)TOTAL * AFFINE_SIZE, nullptr, &err);
	if (err != CL_SUCCESS) { clReleaseMemObject(d_bases); g_ocl_lut_built = true; return; }

	cl_mem d_h_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)TOTAL * FIELD_SIZE, nullptr, &err);
	if (err != CL_SUCCESS) { clReleaseMemObject(d_bases); clReleaseMemObject(d_lut); g_ocl_lut_built = true; return; }

	// Lock the command queue for all GPU dispatches — OpenCL 1.2 queues
	// are not thread-safe, and scan threads may be dispatching concurrently.
	{
		std::lock_guard<std::mutex> lock(g_ocl_mutex);

		// Step 1: compute 16 base points
		clSetKernelArg(g_lut_base_kernel, 0, sizeof(cl_mem), &d_bases);
		size_t one = 1;
		clEnqueueNDRangeKernel(queue, g_lut_base_kernel, 1, nullptr, &one, &one, 0, nullptr, nullptr);
		clFinish(queue);

		// Step 2: fused build + serial inversion (16 work-items)
		clSetKernelArg(g_lut_build_kernel, 0, sizeof(cl_mem), &d_bases);
		clSetKernelArg(g_lut_build_kernel, 1, sizeof(cl_mem), &d_lut);
		clSetKernelArg(g_lut_build_kernel, 2, sizeof(cl_mem), &d_h_buf);
		int n_entries = N;
		clSetKernelArg(g_lut_build_kernel, 3, sizeof(int), &n_entries);
		size_t build_global = SLICES, build_local = 1;
		clEnqueueNDRangeKernel(queue, g_lut_build_kernel, 1, nullptr, &build_global, &build_local, 0, nullptr, nullptr);
		clFinish(queue);

		// Step 3: parallel affine conversion
		clSetKernelArg(g_lut_convert_kernel, 0, sizeof(cl_mem), &d_lut);
		clSetKernelArg(g_lut_convert_kernel, 1, sizeof(cl_mem), &d_h_buf);
		clSetKernelArg(g_lut_convert_kernel, 2, sizeof(int), &n_entries);
		int conv_total = SLICES * (N - 2);
		size_t conv_local = 256;
		size_t conv_global = ((conv_total + conv_local - 1) / conv_local) * conv_local;
		clEnqueueNDRangeKernel(queue, g_lut_convert_kernel, 1, nullptr, &conv_global, &conv_local, 0, nullptr, nullptr);
		clFinish(queue);
	}

	// Cleanup temp buffers
	clReleaseMemObject(d_h_buf);
	clReleaseMemObject(d_bases);

	// Publish (race-safe: only set after table is fully populated)
	g_ocl_gen_lut = d_lut;
	g_ocl_lut_built = true;
	fprintf(stderr, "[OpenCL] Generator LUT built (64 MB)\n");
}

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

static void affine_to_compressed(const ocl::AffinePoint &ap, uint8_t *out33) {
	for (int i = 0; i < 4; i++) {
		uint64_t v = ap.x.limbs[i];
		for (int j = 0; j < 8; j++)
			out33[32 - (i * 8 + j)] = (uint8_t)(v >> (j * 8));
	}
	out33[0] = (ap.y.limbs[0] & 1) ? 0x03 : 0x02;
}

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

			auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());
			auto *queue = static_cast<cl_command_queue>(g_ocl_ctx->native_queue());
			if (ctx && queue) {
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

							// Try to create LUT kernels from the same program
							cl_int lut_err;
							g_lut_base_kernel = clCreateKernel(g_fused_program, "compute_lut_base_points", &lut_err);
							if (lut_err == CL_SUCCESS)
								g_lut_build_kernel = clCreateKernel(g_fused_program, "gen_lut_build_affine_kernel", &lut_err);
							if (lut_err == CL_SUCCESS)
								g_lut_convert_kernel = clCreateKernel(g_fused_program, "gen_lut_convert_zinv_kernel", &lut_err);
							if (lut_err == CL_SUCCESS)
								g_fused_kernel_lut = clCreateKernel(g_fused_program, "bip352_fused_kernel_lut", &lut_err);
							g_use_lut = (lut_err == CL_SUCCESS);
							if (g_use_lut) {
								fprintf(stderr, "[OpenCL] LUT kernels available\n");
							}
						} else {
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
	(void)device_id;

	if (!g_ocl_ctx || !g_ocl_ctx->is_valid())
		return nullptr;

	auto *state = new UfsecpOclBatchState();
	state->count = count;
	state->use_fused = g_use_fused;

	if (state->use_fused) {
		auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());
		cl_int err;

		state->tweak_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * 64,
		                                  const_cast<uint8_t *>(tweak_data), &err);
		if (err != CL_SUCCESS) { delete state; return nullptr; }

		state->scan_key_buf =
		    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, const_cast<uint8_t *>(scan_key), &err);
		if (err != CL_SUCCESS) { clReleaseMemObject(state->tweak_buf); delete state; return nullptr; }

		state->out_x_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, count * 32, nullptr, &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf); clReleaseMemObject(state->scan_key_buf);
			delete state; return nullptr;
		}

		state->out_y_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, count * 32, nullptr, &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf); clReleaseMemObject(state->scan_key_buf);
			clReleaseMemObject(state->out_x_buf); delete state; return nullptr;
		}

		state->midstate_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 8 * sizeof(uint32_t),
		                                     const_cast<uint32_t *>(g_bip352_midstate), &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf); clReleaseMemObject(state->scan_key_buf);
			clReleaseMemObject(state->out_x_buf); clReleaseMemObject(state->out_y_buf);
			delete state; return nullptr;
		}
	} else {
		ocl::Scalar scan_scalar = scalar_from_le(scan_key);
		state->scan_scalars.resize(count, scan_scalar);
		state->tweak_points.resize(count);
		for (uint32_t i = 0; i < count; i++)
			state->tweak_points[i] = affine_from_le(tweak_data + i * 64);
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

		// Build LUT on first use (lazy, thread-safe)
		if (g_use_lut) EnsureOclGenLutBuilt();

		// Lock around setargs + enqueue to prevent concurrent threads from
		// interleaving kernel arguments on the shared cl_kernel object.
		std::lock_guard<std::mutex> lock(g_ocl_mutex);

		cl_kernel kernel;
		if (g_ocl_gen_lut) {
			// LUT path: 7 args
			kernel = g_fused_kernel_lut;
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &state->tweak_buf);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &state->scan_key_buf);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &state->out_x_buf);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &state->out_y_buf);
			clSetKernelArg(kernel, 4, sizeof(cl_mem), &state->midstate_buf);
			clSetKernelArg(kernel, 5, sizeof(cl_mem), &g_ocl_gen_lut);
			clSetKernelArg(kernel, 6, sizeof(uint32_t), &count);
		} else {
			// GLV fallback: 6 args
			kernel = g_fused_kernel;
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &state->tweak_buf);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &state->scan_key_buf);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &state->out_x_buf);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &state->out_y_buf);
			clSetKernelArg(kernel, 4, sizeof(cl_mem), &state->midstate_buf);
			clSetKernelArg(kernel, 5, sizeof(uint32_t), &count);
		}

		size_t local_size = 256;
		clGetKernelWorkGroupInfo(kernel, nullptr, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, nullptr);
		if (local_size > 256) local_size = 256;

		size_t global_size = ((size_t)count + local_size - 1) / local_size * local_size;

		cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
		if (err != CL_SUCCESS) return -1;

		clEnqueueReadBuffer(queue, state->out_x_buf, CL_TRUE, 0, count * 32, out_x, 0, nullptr, nullptr);
		clEnqueueReadBuffer(queue, state->out_y_buf, CL_TRUE, 0, count * 32, out_y, 0, nullptr, nullptr);
		clFinish(queue);

		return 0;
	}

	// ====================================================================
	// Multi-dispatch fallback path
	// ====================================================================
	if (!g_tag_computed) {
		g_tag_midstate = secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");
		g_tag_computed = true;
	}

	std::vector<ocl::JacobianPoint> jac1(count);
	g_ocl_ctx->batch_scalar_mul(state->scan_scalars.data(), state->tweak_points.data(), jac1.data(), count);

	std::vector<ocl::AffinePoint> aff1(count);
	g_ocl_ctx->batch_jacobian_to_affine(jac1.data(), aff1.data(), count);

	std::vector<ocl::Scalar> hash_scalars(count);
	for (uint32_t i = 0; i < count; i++) {
		uint8_t ser[37];
		affine_to_compressed(aff1[i], ser);
		std::memset(ser + 33, 0, 4);
		auto hash = secp256k1::detail::cached_tagged_hash(g_tag_midstate, ser, 37);
		hash_scalars[i] = scalar_from_be(hash.data());
	}

	std::vector<ocl::JacobianPoint> jac2(count);
	g_ocl_ctx->batch_scalar_mul_generator(hash_scalars.data(), jac2.data(), count);

	std::vector<ocl::AffinePoint> aff2(count);
	g_ocl_ctx->batch_jacobian_to_affine(jac2.data(), aff2.data(), count);

	for (uint32_t i = 0; i < count; i++)
		affine_to_le(aff2[i], out_x + i * 32, out_y + i * 32);

	return 0;
}

void UfsecpOclFreeBatch(void *state_handle) {
	if (!state_handle)
		return;
	auto *state = static_cast<UfsecpOclBatchState *>(state_handle);
	if (state->tweak_buf) clReleaseMemObject(state->tweak_buf);
	if (state->scan_key_buf) clReleaseMemObject(state->scan_key_buf);
	if (state->out_x_buf) clReleaseMemObject(state->out_x_buf);
	if (state->out_y_buf) clReleaseMemObject(state->out_y_buf);
	if (state->midstate_buf) clReleaseMemObject(state->midstate_buf);
	delete state;
}

} // extern "C"
