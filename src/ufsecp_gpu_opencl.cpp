// ============================================================================
// ufsecp_gpu_opencl.cpp — BIP-352 GPU pipeline via UltrafastSecp256k1 OpenCL
// ============================================================================

#include "secp256k1_opencl.hpp"
#include <secp256k1/tagged_hash.hpp>
#include <secp256k1/sha256.hpp>
#include <CL/cl.h>

#include <mutex>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "opencl_fused_kernel_source.h"

namespace ocl = secp256k1::opencl;

// ============================================================================
// Global state
// ============================================================================

static std::unique_ptr<ocl::Context> g_ocl_ctx;
static std::mutex g_ocl_mutex;
static bool g_ocl_initialized = false;
static int g_ocl_device_count = 0;

static secp256k1::SHA256 g_tag_midstate;
static bool g_tag_computed = false;

// Kernel handles
static cl_program g_fused_program = nullptr;
static cl_kernel g_fused_kernel = nullptr;
static cl_kernel g_fused_kernel_lut = nullptr;
static cl_kernel g_lut_base_kernel = nullptr;
static cl_kernel g_lut_build_kernel = nullptr;
static cl_kernel g_lut_convert_kernel = nullptr;
static cl_kernel g_full_pass1_kernel = nullptr;
static cl_kernel g_batch_inv_match_kernel = nullptr;
static bool g_use_fused = false;
static bool g_use_lut = false;
static bool g_use_full = false;

// Generator LUT (built once, persistent)
static cl_mem g_ocl_gen_lut = nullptr;
static bool g_ocl_lut_built = false;
static std::mutex g_ocl_lut_mutex;

// Spend key (uploaded once per query)
static cl_mem g_ocl_spend_buf = nullptr;

// ============================================================================
// Generator LUT construction
// ============================================================================

static void EnsureOclGenLutBuilt() {
	if (g_ocl_lut_built)
		return;
	std::lock_guard<std::mutex> lock(g_ocl_lut_mutex);
	if (g_ocl_lut_built)
		return;

	auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());
	auto *queue = static_cast<cl_command_queue>(g_ocl_ctx->native_queue());
	cl_int err;

	static constexpr int LUT_WBITS = 12;
	static constexpr int N = (1 << LUT_WBITS);
	static constexpr int SLICES = (256 + LUT_WBITS - 1) / LUT_WBITS;
	static constexpr int TOTAL = SLICES * N;
	static constexpr size_t AFFINE_SIZE = 64;
	static constexpr size_t FIELD_SIZE = 32;

	cl_mem d_bases = clCreateBuffer(ctx, CL_MEM_READ_WRITE, SLICES * AFFINE_SIZE, nullptr, &err);
	if (err != CL_SUCCESS) {
		g_ocl_lut_built = true;
		return;
	}

	cl_mem d_lut = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)TOTAL * AFFINE_SIZE, nullptr, &err);
	if (err != CL_SUCCESS) {
		clReleaseMemObject(d_bases);
		g_ocl_lut_built = true;
		return;
	}

	cl_mem d_h_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)TOTAL * FIELD_SIZE, nullptr, &err);
	if (err != CL_SUCCESS) {
		clReleaseMemObject(d_bases);
		clReleaseMemObject(d_lut);
		g_ocl_lut_built = true;
		return;
	}

	{
		std::lock_guard<std::mutex> lock(g_ocl_mutex);

		clSetKernelArg(g_lut_base_kernel, 0, sizeof(cl_mem), &d_bases);
		size_t one = 1;
		clEnqueueNDRangeKernel(queue, g_lut_base_kernel, 1, nullptr, &one, &one, 0, nullptr, nullptr);
		clFinish(queue);

		clSetKernelArg(g_lut_build_kernel, 0, sizeof(cl_mem), &d_bases);
		clSetKernelArg(g_lut_build_kernel, 1, sizeof(cl_mem), &d_lut);
		clSetKernelArg(g_lut_build_kernel, 2, sizeof(cl_mem), &d_h_buf);
		int n_entries = N;
		clSetKernelArg(g_lut_build_kernel, 3, sizeof(int), &n_entries);
		size_t build_global = SLICES, build_local = 1;
		clEnqueueNDRangeKernel(queue, g_lut_build_kernel, 1, nullptr, &build_global, &build_local, 0, nullptr, nullptr);
		clFinish(queue);

		clSetKernelArg(g_lut_convert_kernel, 0, sizeof(cl_mem), &d_lut);
		clSetKernelArg(g_lut_convert_kernel, 1, sizeof(cl_mem), &d_h_buf);
		clSetKernelArg(g_lut_convert_kernel, 2, sizeof(int), &n_entries);
		int conv_total = SLICES * (N - 2);
		size_t conv_local = 256;
		size_t conv_global = ((conv_total + conv_local - 1) / conv_local) * conv_local;
		clEnqueueNDRangeKernel(queue, g_lut_convert_kernel, 1, nullptr, &conv_global, &conv_local, 0, nullptr, nullptr);
		clFinish(queue);
	}

	clReleaseMemObject(d_h_buf);
	clReleaseMemObject(d_bases);

	g_ocl_gen_lut = d_lut;
	g_ocl_lut_built = true;
	fprintf(stderr, "[OpenCL] Generator LUT built (%d MB)\n", (int)((size_t)TOTAL * AFFINE_SIZE / (1024 * 1024)));
}

// ============================================================================
// Per-batch state
// ============================================================================

struct UfsecpOclBatchState {
	cl_mem tweak_buf;
	cl_mem scan_plan_buf;
	// Legacy (phases 1-4)
	cl_mem out_x_buf;
	cl_mem out_y_buf;
	// Full pipeline (phases 1-6)
	cl_mem output_prefixes_buf;
	cl_mem output_offsets_buf;
	cl_mem output_lengths_buf;
	cl_mem match_flags_buf;
	cl_mem cand_x_buf;
	cl_mem cand_z_buf;
	cl_mem output_pts_buf;
	// Multi-dispatch fallback
	std::vector<ocl::Scalar> scan_scalars;
	std::vector<ocl::AffinePoint> tweak_points;
	uint32_t count;
	bool use_fused;
	bool full_pipeline;
};

// ============================================================================
// Byte-order helpers (multi-dispatch fallback)
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
// Extern "C" interface
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

							cl_int lut_err;
							g_lut_base_kernel = clCreateKernel(g_fused_program, "compute_lut_base_points", &lut_err);
							if (lut_err == CL_SUCCESS)
								g_lut_build_kernel =
								    clCreateKernel(g_fused_program, "gen_lut_build_affine_kernel", &lut_err);
							if (lut_err == CL_SUCCESS)
								g_lut_convert_kernel =
								    clCreateKernel(g_fused_program, "gen_lut_convert_zinv_kernel", &lut_err);
							if (lut_err == CL_SUCCESS)
								g_fused_kernel_lut =
								    clCreateKernel(g_fused_program, "bip352_fused_kernel_lut", &lut_err);
							g_use_lut = (lut_err == CL_SUCCESS);
							if (g_use_lut)
								fprintf(stderr, "[OpenCL] LUT kernels available\n");

							// Full pipeline kernels
							cl_int fp_err;
							g_full_pass1_kernel = clCreateKernel(g_fused_program, "bip352_full_pass1", &fp_err);
							if (fp_err == CL_SUCCESS)
								g_batch_inv_match_kernel =
								    clCreateKernel(g_fused_program, "bip352_batch_inv_match", &fp_err);
							g_use_full = (fp_err == CL_SUCCESS && g_use_lut);
							if (g_use_full)
								fprintf(stderr, "[OpenCL] Full pipeline kernels available\n");
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

// Upload spend key + labels (once per query)
void UfsecpOclSetSpendKey(const uint8_t *spend_xy, int num_labels, const uint8_t *label_keys_xy, int device_id) {
	(void)device_id;
	if (!g_ocl_ctx)
		return;

	auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());

	// BIP352SpendKeys: AffinePoint base + AffinePoint[16] labels + u8 num_labels + u8[3] pad
	// AffinePoint = 64 bytes, total = 64 + 16*64 + 4 = 1092 bytes
	static constexpr size_t SPEND_SIZE = 64 + 16 * 64 + 4;
	uint8_t spend_data[SPEND_SIZE] = {};

	// base (64 LE bytes, already in wire format)
	std::memcpy(spend_data, spend_xy, 64);

	int n = (num_labels > 16) ? 16 : num_labels;
	if (label_keys_xy && n > 0)
		std::memcpy(spend_data + 64, label_keys_xy, n * 64);

	spend_data[64 + 16 * 64] = (uint8_t)n; // num_labels

	if (g_ocl_spend_buf)
		clReleaseMemObject(g_ocl_spend_buf);
	cl_int err;
	g_ocl_spend_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SPEND_SIZE, spend_data, &err);
}

// Full pipeline launch
void *UfsecpOclLaunchBatchFull(const uint8_t *tweak_data, const int64_t *output_prefixes, uint32_t total_outputs,
                               const uint32_t *output_offsets, const uint8_t *output_lengths, uint32_t count,
                               int device_id, const void *precomp) {
	(void)device_id;
	if (!g_ocl_ctx || !g_ocl_ctx->is_valid() || !g_use_full)
		return nullptr;

	auto *ctx_cl = static_cast<cl_context>(g_ocl_ctx->native_context());
	cl_int err;

	auto *state = new UfsecpOclBatchState();
	state->count = count;
	state->use_fused = true;
	state->full_pipeline = true;
	state->out_x_buf = nullptr;
	state->out_y_buf = nullptr;

	state->tweak_buf = clCreateBuffer(ctx_cl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * 64,
	                                  const_cast<uint8_t *>(tweak_data), &err);
	if (err != CL_SUCCESS) {
		delete state;
		return nullptr;
	}

	state->scan_plan_buf =
	    clCreateBuffer(ctx_cl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 264, const_cast<void *>(precomp), &err);
	if (err != CL_SUCCESS)
		goto fail;

	state->output_prefixes_buf =
	    clCreateBuffer(ctx_cl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)total_outputs * sizeof(int64_t),
	                   const_cast<int64_t *>(output_prefixes), &err);
	if (err != CL_SUCCESS)
		goto fail;

	state->output_offsets_buf =
	    clCreateBuffer(ctx_cl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)count * sizeof(uint32_t),
	                   const_cast<uint32_t *>(output_offsets), &err);
	if (err != CL_SUCCESS)
		goto fail;

	state->output_lengths_buf = clCreateBuffer(ctx_cl, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)count,
	                                           const_cast<uint8_t *>(output_lengths), &err);
	if (err != CL_SUCCESS)
		goto fail;

	state->match_flags_buf = clCreateBuffer(ctx_cl, CL_MEM_WRITE_ONLY, (size_t)count, nullptr, &err);
	if (err != CL_SUCCESS)
		goto fail;

	// FieldElement = 32 bytes, JacobianPoint = 4*32 + 4 padding... check OpenCL struct size
	// In OpenCL: JacobianPoint = {FieldElement x, y, z; uint infinity; } = 3*32 + 4 = 100, but
	// the library may pad differently. Use sizeof from the host types.
	state->cand_x_buf = clCreateBuffer(ctx_cl, CL_MEM_READ_WRITE, (size_t)count * 32, nullptr, &err);
	if (err != CL_SUCCESS)
		goto fail;

	state->cand_z_buf = clCreateBuffer(ctx_cl, CL_MEM_READ_WRITE, (size_t)count * 32, nullptr, &err);
	if (err != CL_SUCCESS)
		goto fail;

	// JacobianPoint in OpenCL: {FieldElement x, y, z; uint infinity;} with padding
	// FieldElement = 4 * ulong = 32 bytes. JacobianPoint = 3*32 + 4 = 100, but likely padded to 128
	// Let's use 128 bytes to be safe (matches CUDA alignment)
	state->output_pts_buf = clCreateBuffer(ctx_cl, CL_MEM_READ_WRITE, (size_t)count * 128, nullptr, &err);
	if (err != CL_SUCCESS)
		goto fail;

	return state;

fail:
	if (state->tweak_buf)
		clReleaseMemObject(state->tweak_buf);
	if (state->scan_plan_buf)
		clReleaseMemObject(state->scan_plan_buf);
	if (state->output_prefixes_buf)
		clReleaseMemObject(state->output_prefixes_buf);
	if (state->output_offsets_buf)
		clReleaseMemObject(state->output_offsets_buf);
	if (state->output_lengths_buf)
		clReleaseMemObject(state->output_lengths_buf);
	if (state->match_flags_buf)
		clReleaseMemObject(state->match_flags_buf);
	if (state->cand_x_buf)
		clReleaseMemObject(state->cand_x_buf);
	if (state->cand_z_buf)
		clReleaseMemObject(state->cand_z_buf);
	if (state->output_pts_buf)
		clReleaseMemObject(state->output_pts_buf);
	delete state;
	return nullptr;
}

int UfsecpOclRunKernelsFull(void *state_handle, uint8_t *match_flags, uint32_t count) {
	auto *state = static_cast<UfsecpOclBatchState *>(state_handle);
	if (!state || !g_ocl_ctx || !g_ocl_gen_lut || !g_ocl_spend_buf)
		return -1;

	auto *queue = static_cast<cl_command_queue>(g_ocl_ctx->native_queue());

	if (g_use_lut)
		EnsureOclGenLutBuilt();
	if (!g_ocl_gen_lut)
		return -1;

	std::lock_guard<std::mutex> lock(g_ocl_mutex);

	// Pass 1: phases 1-4 + add spend key
	{
		cl_kernel k = g_full_pass1_kernel;
		clSetKernelArg(k, 0, sizeof(cl_mem), &state->tweak_buf);
		clSetKernelArg(k, 1, sizeof(cl_mem), &state->scan_plan_buf);
		clSetKernelArg(k, 2, sizeof(cl_mem), &g_ocl_gen_lut);
		clSetKernelArg(k, 3, sizeof(cl_mem), &g_ocl_spend_buf);
		clSetKernelArg(k, 4, sizeof(cl_mem), &state->cand_x_buf);
		clSetKernelArg(k, 5, sizeof(cl_mem), &state->cand_z_buf);
		clSetKernelArg(k, 6, sizeof(cl_mem), &state->output_pts_buf);
		clSetKernelArg(k, 7, sizeof(uint32_t), &count);

		size_t local = 128, global = ((count + local - 1) / local) * local;
		cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
		if (err != CL_SUCCESS)
			return -1;
	}

	// Pass 2: fused batch inversion + match
	{
		cl_kernel k = g_batch_inv_match_kernel;
		clSetKernelArg(k, 0, sizeof(cl_mem), &state->cand_x_buf);
		clSetKernelArg(k, 1, sizeof(cl_mem), &state->cand_z_buf);
		clSetKernelArg(k, 2, sizeof(cl_mem), &state->output_pts_buf);
		clSetKernelArg(k, 3, sizeof(cl_mem), &g_ocl_spend_buf);
		clSetKernelArg(k, 4, sizeof(cl_mem), &state->output_prefixes_buf);
		clSetKernelArg(k, 5, sizeof(cl_mem), &state->output_offsets_buf);
		clSetKernelArg(k, 6, sizeof(cl_mem), &state->output_lengths_buf);
		clSetKernelArg(k, 7, sizeof(cl_mem), &state->match_flags_buf);
		// local memory: 2 * local_size * sizeof(FieldElement)
		size_t local = 256;
		size_t shared_size = 2 * local * 32; // FieldElement = 32 bytes
		clSetKernelArg(k, 8, shared_size, nullptr);
		clSetKernelArg(k, 9, sizeof(uint32_t), &count);

		size_t global = ((count + local - 1) / local) * local;
		cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
		if (err != CL_SUCCESS)
			return -1;
	}

	clEnqueueReadBuffer(queue, state->match_flags_buf, CL_TRUE, 0, count, match_flags, 0, nullptr, nullptr);
	clFinish(queue);

	return 0;
}

// Legacy interface (phases 1-4 only)
void *UfsecpOclLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id,
                           const void *precomp) {
	(void)device_id;
	if (!g_ocl_ctx || !g_ocl_ctx->is_valid())
		return nullptr;

	auto *state = new UfsecpOclBatchState();
	state->count = count;
	state->use_fused = g_use_fused;
	state->full_pipeline = false;
	state->output_prefixes_buf = nullptr;
	state->output_offsets_buf = nullptr;
	state->output_lengths_buf = nullptr;
	state->match_flags_buf = nullptr;
	state->cand_x_buf = nullptr;
	state->cand_z_buf = nullptr;
	state->output_pts_buf = nullptr;

	if (state->use_fused) {
		auto *ctx = static_cast<cl_context>(g_ocl_ctx->native_context());
		cl_int err;

		state->tweak_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * 64,
		                                  const_cast<uint8_t *>(tweak_data), &err);
		if (err != CL_SUCCESS) {
			delete state;
			return nullptr;
		}

		state->scan_plan_buf =
		    clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 264, const_cast<void *>(precomp), &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf);
			delete state;
			return nullptr;
		}

		state->out_x_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, count * 32, nullptr, &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf);
			clReleaseMemObject(state->scan_plan_buf);
			delete state;
			return nullptr;
		}

		state->out_y_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, count * 32, nullptr, &err);
		if (err != CL_SUCCESS) {
			clReleaseMemObject(state->tweak_buf);
			clReleaseMemObject(state->scan_plan_buf);
			clReleaseMemObject(state->out_x_buf);
			delete state;
			return nullptr;
		}
	} else {
		ocl::Scalar scan_scalar = scalar_from_le(scan_key);
		state->scan_scalars.resize(count, scan_scalar);
		state->tweak_points.resize(count);
		for (uint32_t i = 0; i < count; i++)
			state->tweak_points[i] = affine_from_le(tweak_data + i * 64);
		state->tweak_buf = nullptr;
		state->scan_plan_buf = nullptr;
		state->out_x_buf = nullptr;
		state->out_y_buf = nullptr;
	}

	return state;
}

int UfsecpOclRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count) {
	auto *state = static_cast<UfsecpOclBatchState *>(state_handle);
	if (!state || !g_ocl_ctx)
		return -1;

	if (state->use_fused) {
		auto *queue = static_cast<cl_command_queue>(g_ocl_ctx->native_queue());
		if (g_use_lut)
			EnsureOclGenLutBuilt();

		std::lock_guard<std::mutex> lock(g_ocl_mutex);

		cl_kernel kernel;
		if (g_ocl_gen_lut) {
			kernel = g_fused_kernel_lut;
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &state->tweak_buf);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &state->scan_plan_buf);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &state->out_x_buf);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &state->out_y_buf);
			clSetKernelArg(kernel, 4, sizeof(cl_mem), &g_ocl_gen_lut);
			clSetKernelArg(kernel, 5, sizeof(uint32_t), &count);
		} else {
			kernel = g_fused_kernel;
			clSetKernelArg(kernel, 0, sizeof(cl_mem), &state->tweak_buf);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &state->scan_plan_buf);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &state->out_x_buf);
			clSetKernelArg(kernel, 3, sizeof(cl_mem), &state->out_y_buf);
			clSetKernelArg(kernel, 4, sizeof(uint32_t), &count);
		}

		size_t local = 256, global = ((count + local - 1) / local) * local;
		cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
		if (err != CL_SUCCESS)
			return -1;

		clEnqueueReadBuffer(queue, state->out_x_buf, CL_TRUE, 0, count * 32, out_x, 0, nullptr, nullptr);
		clEnqueueReadBuffer(queue, state->out_y_buf, CL_TRUE, 0, count * 32, out_y, 0, nullptr, nullptr);
		clFinish(queue);
		return 0;
	}

	// Multi-dispatch fallback
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
	if (state->tweak_buf)
		clReleaseMemObject(state->tweak_buf);
	if (state->scan_plan_buf)
		clReleaseMemObject(state->scan_plan_buf);
	if (state->out_x_buf)
		clReleaseMemObject(state->out_x_buf);
	if (state->out_y_buf)
		clReleaseMemObject(state->out_y_buf);
	if (state->output_prefixes_buf)
		clReleaseMemObject(state->output_prefixes_buf);
	if (state->output_offsets_buf)
		clReleaseMemObject(state->output_offsets_buf);
	if (state->output_lengths_buf)
		clReleaseMemObject(state->output_lengths_buf);
	if (state->match_flags_buf)
		clReleaseMemObject(state->match_flags_buf);
	if (state->cand_x_buf)
		clReleaseMemObject(state->cand_x_buf);
	if (state->cand_z_buf)
		clReleaseMemObject(state->cand_z_buf);
	if (state->output_pts_buf)
		clReleaseMemObject(state->output_pts_buf);
	delete state;
}

} // extern "C"
