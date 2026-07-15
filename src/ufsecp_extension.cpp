#define DUCKDB_EXTENSION_MAIN

#include "ufsecp_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

// UltrafastSecp256k1 CPU headers
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/field.hpp>
#include <secp256k1/tagged_hash.hpp>
#include <secp256k1/sha256.hpp>
#include <secp256k1/batch_add_affine.hpp>
#include <secp256k1/precompute.hpp>

#include <vector>
#include <cstring>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <memory>

// ============================================================================
// Progress registry — side-channel progress reporting for ufsecp_progress()
// ============================================================================

struct ScanProgress {
	std::atomic<uint64_t> rows_received {0};
	std::atomic<uint64_t> rows_processed {0};
	// Optional caller-supplied row total for the input subquery (set via the
	// `total_rows` named parameter on ufsecp_scan). When set (>0),
	// ufsecp_progress reports `received / total_rows`, which advances
	// per-chunk (smooth) instead of per-batch (granular). When 0, falls
	// back to processed/received, which jumps in batch_size increments.
	uint64_t total_rows {0};
	// `complete` is consulted by ufsecp_progress() only in the empty-scan
	// case (received stays 0). For non-empty scans, total_rows (if set) or
	// processed/received is the authoritative progress signal.
	std::atomic<bool> complete {false};
};

static std::mutex g_progress_mutex;
static std::unordered_map<std::string, std::shared_ptr<ScanProgress>> g_progress_map;

// Conditional GPU support — extern "C" declarations for each backend
#ifdef UFSECP_CUDA_ENABLED
extern "C" {
int UfsecpCudaDetect(int *num_gpus);
void *UfsecpCudaLaunchBatchFull(const uint8_t *tweak_data, const int64_t *output_prefixes, uint32_t total_outputs,
                                const uint32_t *output_offsets, const uint8_t *output_lengths, uint32_t count,
                                int device_id, const void *precomp, const uint8_t *spend_xy, int num_labels,
                                const uint8_t *label_keys_xy);
int UfsecpCudaRunKernelsFull(void *state_handle, uint8_t *match_flags, uint32_t count);
void *UfsecpCudaLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id,
                            const void *precomp);
int UfsecpCudaRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count);
void UfsecpCudaFreeBatch(void *state_handle);
}
#endif
#ifdef UFSECP_OPENCL_ENABLED
extern "C" {
int UfsecpOclDetect(int *num_gpus);
void UfsecpOclEnsureReady();
void *UfsecpOclLaunchBatchFull(const uint8_t *tweak_data, const int64_t *output_prefixes, uint32_t total_outputs,
                               const uint32_t *output_offsets, const uint8_t *output_lengths, uint32_t count,
                               int device_id, const void *precomp, const uint8_t *spend_xy, int num_labels,
                               const uint8_t *label_keys_xy);
int UfsecpOclRunKernelsFull(void *state_handle, uint8_t *match_flags, uint32_t count);
void *UfsecpOclLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id,
                           const void *precomp);
int UfsecpOclRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count);
void UfsecpOclFreeBatch(void *state_handle);
}
#endif
#ifdef UFSECP_METAL_ENABLED
extern "C" {
int UfsecpMetalDetect(int *num_gpus);
void UfsecpMetalEnsureReady();
void *UfsecpMetalLaunchBatchFull(const uint8_t *tweak_data, const int64_t *output_prefixes, uint32_t total_outputs,
                                 const uint32_t *output_offsets, const uint8_t *output_lengths, uint32_t count,
                                 int device_id, const void *precomp, const uint8_t *spend_xy, int num_labels,
                                 const uint8_t *label_keys_xy);
int UfsecpMetalRunKernelsFull(void *state_handle, uint8_t *match_flags, uint32_t count);
void *UfsecpMetalLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id,
                             const void *precomp);
int UfsecpMetalRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count);
void UfsecpMetalFreeBatch(void *state_handle);
}
#endif

namespace duckdb {

using secp256k1::fast::AffinePointCompact;
using secp256k1::fast::FieldElement;
using secp256k1::fast::KPlan;
using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

// ============================================================================
// GPU detection state (compile-time conditional)
// ============================================================================

#ifdef UFSECP_GPU_ENABLED
enum class GpuBackend { NONE, CUDA, OPENCL, METAL };

static GpuBackend g_gpu_backend = GpuBackend::NONE;
static int g_num_gpus = 0;
static bool g_gpu_detected = false;
static std::mutex g_gpu_init_mutex;

// Function pointers for the active GPU backend — full pipeline (phases 1-6).
// Spend bytes are passed inline through LaunchBatchFull (per-batch); LUT-style
// state is built eagerly via EnsureReady (called once per Bind, before any
// worker thread runs).
static void (*g_gpu_ensure_ready)() = nullptr;
static void *(*g_gpu_launch_full)(const uint8_t *, const int64_t *, uint32_t, const uint32_t *, const uint8_t *,
                                  uint32_t, int, const void *, const uint8_t *, int, const uint8_t *) = nullptr;
static int (*g_gpu_run_full)(void *, uint8_t *, uint32_t) = nullptr;

// Function pointers for the active GPU backend — legacy (phases 1-4, fallback)
static void *(*g_gpu_launch)(const uint8_t *, const uint8_t *, uint32_t, int, const void *) = nullptr;
static int (*g_gpu_run)(void *, uint8_t *, uint8_t *, uint32_t) = nullptr;
static void (*g_gpu_free)(void *) = nullptr;

static void EnsureGpuDetected() {
	if (g_gpu_detected)
		return;
	std::lock_guard<std::mutex> lock(g_gpu_init_mutex);
	if (g_gpu_detected)
		return;

		// Try CUDA first (higher performance), then OpenCL
#ifdef UFSECP_CUDA_ENABLED
	{
		int cuda_gpus = 0;
		UfsecpCudaDetect(&cuda_gpus);
		if (cuda_gpus > 0) {
			g_num_gpus = cuda_gpus;
			g_gpu_backend = GpuBackend::CUDA;
			g_gpu_launch_full = UfsecpCudaLaunchBatchFull;
			g_gpu_run_full = UfsecpCudaRunKernelsFull;
			g_gpu_launch = UfsecpCudaLaunchBatch;
			g_gpu_run = UfsecpCudaRunKernels;
			g_gpu_free = UfsecpCudaFreeBatch;
			g_gpu_detected = true;
			return;
		}
	}
#endif
#ifdef UFSECP_OPENCL_ENABLED
	{
		int ocl_gpus = 0;
		UfsecpOclDetect(&ocl_gpus);
		if (ocl_gpus > 0) {
			g_num_gpus = ocl_gpus;
			g_gpu_backend = GpuBackend::OPENCL;
			g_gpu_ensure_ready = UfsecpOclEnsureReady;
			g_gpu_launch_full = UfsecpOclLaunchBatchFull;
			g_gpu_run_full = UfsecpOclRunKernelsFull;
			g_gpu_launch = UfsecpOclLaunchBatch;
			g_gpu_run = UfsecpOclRunKernels;
			g_gpu_free = UfsecpOclFreeBatch;
			g_gpu_detected = true;
			return;
		}
	}
#endif
#ifdef UFSECP_METAL_ENABLED
	{
		int metal_gpus = 0;
		UfsecpMetalDetect(&metal_gpus);
		if (metal_gpus > 0) {
			g_num_gpus = metal_gpus;
			g_gpu_backend = GpuBackend::METAL;
			g_gpu_ensure_ready = UfsecpMetalEnsureReady;
			g_gpu_launch_full = UfsecpMetalLaunchBatchFull;
			g_gpu_run_full = UfsecpMetalRunKernelsFull;
			g_gpu_launch = UfsecpMetalLaunchBatch;
			g_gpu_run = UfsecpMetalRunKernels;
			g_gpu_free = UfsecpMetalFreeBatch;
			g_gpu_detected = true;
			return;
		}
	}
#endif
	g_gpu_detected = true;
}
#endif

// ============================================================================
// Data format conversion helpers
// ============================================================================

// Convert 32 little-endian bytes (Frigate wire format) to a FieldElement.
// Frigate's getSecp256k1PubKey() produces 64-byte keys as x_LE || y_LE.
// UltrafastSecp256k1's from_bytes() expects big-endian.
static FieldElement FieldElementFromLE(const uint8_t *le_bytes) {
	std::array<uint8_t, 32> be;
	for (int i = 0; i < 32; i++) {
		be[i] = le_bytes[31 - i];
	}
	return FieldElement::from_bytes(be);
}

// Convert 32 little-endian bytes to a Scalar.
// Frigate's scan_private_key is sent as Utils.reverseBytes(privKeyBytes).
static Scalar ScalarFromLE(const uint8_t *le_bytes) {
	std::array<uint8_t, 32> be;
	for (int i = 0; i < 32; i++) {
		be[i] = le_bytes[31 - i];
	}
	return Scalar::from_bytes(be);
}

// Extract upper 64 bits of a FieldElement as int64_t.
// Matches cudasp convention: digits[6] | (digits[7] << 32) where digits
// are LE u32 limbs — equivalent to the most-significant 8 bytes in big-endian.
static int64_t ExtractUpper64(const FieldElement &fe) {
	uint8_t bytes[32];
	fe.to_bytes_into(bytes); // big-endian, no allocation
	uint64_t value = 0;
	for (int i = 0; i < 8; i++) {
		value = (value << 8) | bytes[i];
	}
	return static_cast<int64_t>(value);
}

// ============================================================================
// Bind data — precomputed query constants
// ============================================================================

struct UfsecpScanBindData : public TableFunctionData {
	UfsecpScanBindData() : batch_size(300000), total_rows(0) {
	}
	~UfsecpScanBindData() {
		if (progress) {
			// Only erase the map entry if it's still pointing at OUR ScanProgress.
			// Concurrent scans with the same scan_key share a key in the map; we
			// don't want to erase an entry that belongs to a peer scan. See the
			// matching insertion logic in UfsecpScanBind.
			std::lock_guard<std::mutex> lock(g_progress_mutex);
			auto it = g_progress_map.find(scan_private_key_data);
			if (it != g_progress_map.end() && it->second == progress) {
				g_progress_map.erase(it);
			}
		}
	}

	static constexpr idx_t TWEAK_KEY_SIZE = 64; // 64 bytes: uncompressed EC point (32-byte x || 32-byte y)
	static constexpr idx_t SCALAR_SIZE = 32;    // 32 bytes: scalar for EC multiplication

	idx_t batch_size;
	// Caller-supplied row count for the input subquery (named parameter
	// total_rows, default 0 = "not provided"). When set, ufsecp_progress
	// reports received/total_rows for smooth per-chunk granularity. The
	// caller should pass `SELECT COUNT(*) FROM <same_input>` — i.e. with
	// the same WHERE filter that the scan's input subquery uses.
	idx_t total_rows;

	// Precomputed at bind time from scan_private_key
	KPlan kplan;

	// Precomputed tagged hash midstate for "BIP0352/SharedSecret"
	secp256k1::SHA256 tag_midstate;

	// Spend public key (affine coordinates)
	FieldElement spend_x;
	FieldElement spend_y;

	// Precomputed labelled spend keys: spend_public_key + label_key[L]
	// For the base case (no label), spend_x/spend_y is used directly.
	std::vector<AffinePointCompact> labelled_spend_keys;

	// Raw copies for validation (same pattern as cudasp)
	std::string scan_private_key_data;
	std::string spend_public_key_data;
	std::vector<std::string> label_keys_data;

	// Backend selection: "cpu", "gpu", or "auto" (default)
	std::string backend = "auto";
	bool use_gpu = false; // resolved at bind time from backend + GPU detection

	// Progress tracking (shared with g_progress_map for side-channel polling)
	std::shared_ptr<ScanProgress> progress;
};

// ============================================================================
// Local state — per-thread accumulation buffers
// ============================================================================

struct UfsecpScanLocalState : public LocalTableFunctionState {
	UfsecpScanLocalState() : finalized(false), match_position(0) {
	}

	bool finalized;

	// Per-thread accumulated input data (same layout as cudasp)
	vector<std::string> accumulated_txids;
	vector<int32_t> accumulated_heights;
	vector<std::string> accumulated_tweak_keys;
	vector<int64_t> accumulated_outputs;
	vector<idx_t> accumulated_output_offsets;
	vector<idx_t> accumulated_output_lengths;

	// Per-thread match output (emitted in chunks from this thread's Finalize).
	// Storing per-thread instead of in a single global avoids the cross-thread
	// race where a no-data thread becomes the "output thread" before the
	// data-having threads have populated their matches.
	vector<std::string> match_txids;
	vector<int32_t> match_heights;
	vector<std::string> match_tweak_keys;
	idx_t match_position;

	// Reusable scratch buffers (avoid per-batch heap allocation)
	std::vector<FieldElement> scratch;

#ifdef UFSECP_GPU_ENABLED
	int assigned_gpu = -1; // GPU device ID for this thread (-1 = CPU)
#endif
};

// ============================================================================
// Global state — thread synchronization and output collection
// ============================================================================

struct UfsecpScanState : public GlobalTableFunctionState {
	UfsecpScanState() {
	}

	// Since DuckDB 1.5.1 (PR #19951), in/out function pipeline parallelism is capped by
	// this value; the base class default of 1 forces single-threaded execution.
	idx_t MaxThreads() const override {
		return GlobalTableFunctionState::MAX_THREADS;
	}
};

// ============================================================================
// AccumulateInput — copy input chunk into per-thread owned storage
// ============================================================================

static void AccumulateInput(UfsecpScanLocalState &local_state, DataChunk &input) {
	idx_t count = input.size();

	// Expected columns: txid (BLOB), height (INTEGER), tweak_key (BLOB), outputs (LIST[BIGINT])
	auto &txid_column = input.data[0];
	auto &height_column = input.data[1];
	auto &tweak_key_column = input.data[2];
	auto &outputs_column = input.data[3];

	UnifiedVectorFormat txid_data;
	UnifiedVectorFormat height_data;
	UnifiedVectorFormat tweak_key_data;
	UnifiedVectorFormat outputs_data;

	txid_column.ToUnifiedFormat(count, txid_data);
	height_column.ToUnifiedFormat(count, height_data);
	tweak_key_column.ToUnifiedFormat(count, tweak_key_data);
	outputs_column.ToUnifiedFormat(count, outputs_data);

	auto txid_ptr = UnifiedVectorFormat::GetData<string_t>(txid_data);
	auto height_ptr = UnifiedVectorFormat::GetData<int32_t>(height_data);
	auto tweak_key_ptr = UnifiedVectorFormat::GetData<string_t>(tweak_key_data);
	auto outputs_entries = UnifiedVectorFormat::GetData<list_entry_t>(outputs_data);

	auto &outputs_child = ListVector::GetEntry(outputs_column);
	UnifiedVectorFormat outputs_child_data;
	outputs_child.ToUnifiedFormat(ListVector::GetListSize(outputs_column), outputs_child_data);
	auto outputs_child_ptr = UnifiedVectorFormat::GetData<int64_t>(outputs_child_data);

	for (idx_t i = 0; i < count; i++) {
		auto txid_idx = txid_data.sel->get_index(i);
		auto height_idx = height_data.sel->get_index(i);
		auto tweak_key_idx = tweak_key_data.sel->get_index(i);
		auto outputs_idx = outputs_data.sel->get_index(i);

		if (txid_data.validity.RowIsValid(txid_idx) && height_data.validity.RowIsValid(height_idx) &&
		    tweak_key_data.validity.RowIsValid(tweak_key_idx)) {
			auto txid_str = txid_ptr[txid_idx];
			auto tweak_key_str = tweak_key_ptr[tweak_key_idx];
			local_state.accumulated_txids.push_back(std::string(txid_str.GetData(), txid_str.GetSize()));
			local_state.accumulated_heights.push_back(height_ptr[height_idx]);
			local_state.accumulated_tweak_keys.push_back(std::string(tweak_key_str.GetData(), tweak_key_str.GetSize()));

			idx_t outputs_offset = local_state.accumulated_outputs.size();
			local_state.accumulated_output_offsets.push_back(outputs_offset);

			idx_t outputs_len = 0;
			if (outputs_data.validity.RowIsValid(outputs_idx)) {
				auto &outputs_entry = outputs_entries[outputs_idx];
				for (idx_t out_i = 0; out_i < outputs_entry.length; out_i++) {
					auto child_idx = outputs_child_data.sel->get_index(outputs_entry.offset + out_i);
					if (outputs_child_data.validity.RowIsValid(child_idx)) {
						local_state.accumulated_outputs.push_back(outputs_child_ptr[child_idx]);
						outputs_len++;
					}
				}
			}
			local_state.accumulated_output_lengths.push_back(outputs_len);
		}
	}
}

// ============================================================================
// ProcessBatch — BIP-352 Silent Payments scanning pipeline (Option C)
// ============================================================================
//
// Pipeline phases:
//   1. Per-row: EC scalar multiply  — shared_secret = tweak_key × scan_private_key
//   2. Per-row: Serialize compressed SEC1 + 4 zero bytes (k=0)
//   3. Per-row: Tagged hash SHA256("BIP0352/SharedSecret", serialized)
//   4. Per-row: Generator multiply  — output_point = hash × G
//   5. Batch:   Z-inversion         — Jacobian → affine via fe_batch_inverse
//   6. Batch:   Point addition      — final = output_point + spend_key
//   7. Per-row: Match check         — upper 64 bits of x vs outputs
//   8. Batch:   Label addition      — labelled = output_point + labelled_spend_key[L]
//

static void ProcessBatch(UfsecpScanLocalState &local_state, const UfsecpScanBindData &bind_data,
                         UfsecpScanState &global_state) {
	idx_t N = local_state.accumulated_txids.size();
	if (N == 0)
		return;

	// ================================================================
	// Phase 1: Per-row EC operations (steps 1-4)
	// ================================================================
	// Store Jacobian coordinates separately for batch Z-inversion.
	std::vector<FieldElement> jac_X(N);
	std::vector<FieldElement> jac_Y(N);
	std::vector<FieldElement> jac_Z(N);

	for (idx_t i = 0; i < N; i++) {
		const uint8_t *tweak_data = reinterpret_cast<const uint8_t *>(local_state.accumulated_tweak_keys[i].data());

		// Step 1: shared_secret = tweak_key × scan_private_key
		FieldElement tweak_x = FieldElementFromLE(tweak_data);
		FieldElement tweak_y = FieldElementFromLE(tweak_data + 32);
		Point tweak_point = Point::from_affine(tweak_x, tweak_y);
		Point shared_secret = tweak_point.scalar_mul_with_plan(bind_data.kplan);

		// Step 2: Compressed SEC1 serialization + 4 zero bytes (output index k=0)
		auto compressed = shared_secret.to_compressed(); // 33 bytes: 0x02|0x03 || x
		uint8_t serialized[37];
		std::memcpy(serialized, compressed.data(), 33);
		std::memset(serialized + 33, 0, 4);

		// Step 3: Tagged hash with precomputed midstate
		auto hash = secp256k1::detail::cached_tagged_hash(bind_data.tag_midstate, serialized, 37);

		// Step 4: output_point = hash × G (generator multiplication)
		Scalar hash_scalar = Scalar::from_bytes(hash.data());
		Point output_point = Point::generator().scalar_mul(hash_scalar);

		// Store Jacobian coordinates for batch conversion
		jac_X[i] = output_point.X();
		jac_Y[i] = output_point.Y();
		jac_Z[i] = output_point.z();
	}

	// ================================================================
	// Phase 2: Batch Z-inversion (Jacobian → affine)
	// ================================================================
	// Montgomery batch inversion: 1 inverse + 3(N-1) muls ≈ 70 ns/point
	// vs N individual inversions at ~3 µs each.
	secp256k1::fast::fe_batch_inverse(jac_Z.data(), N, local_state.scratch);

	std::vector<AffinePointCompact> offsets(N);
	for (idx_t i = 0; i < N; i++) {
		FieldElement z_inv_sq = jac_Z[i] * jac_Z[i];
		FieldElement z_inv_cu = z_inv_sq * jac_Z[i];
		offsets[i].x = jac_X[i] * z_inv_sq;
		offsets[i].y = jac_Y[i] * z_inv_cu;
	}

	// ================================================================
	// Phase 3: Batch addition — base case (spend_key + output_point[i])
	// ================================================================
	std::vector<FieldElement> final_x(N);
	secp256k1::fast::batch_add_affine_x(bind_data.spend_x, bind_data.spend_y, offsets.data(), final_x.data(), N,
	                                    local_state.scratch);

	// ================================================================
	// Phase 4: Match checking — base case
	// ================================================================
	std::vector<bool> matched(N, false);
	for (idx_t i = 0; i < N; i++) {
		int64_t upper64 = ExtractUpper64(final_x[i]);
		idx_t off = local_state.accumulated_output_offsets[i];
		idx_t len = local_state.accumulated_output_lengths[i];
		for (idx_t j = 0; j < len; j++) {
			if (local_state.accumulated_outputs[off + j] == upper64) {
				matched[i] = true;
				break;
			}
		}
	}

	// ================================================================
	// Phase 5: Batch addition + match checking — label cases
	// ================================================================
	for (idx_t L = 0; L < bind_data.labelled_spend_keys.size(); L++) {
		std::vector<FieldElement> labelled_x(N);
		secp256k1::fast::batch_add_affine_x(bind_data.labelled_spend_keys[L].x, bind_data.labelled_spend_keys[L].y,
		                                    offsets.data(), labelled_x.data(), N, local_state.scratch);

		for (idx_t i = 0; i < N; i++) {
			if (matched[i])
				continue; // already matched
			int64_t upper64 = ExtractUpper64(labelled_x[i]);
			idx_t off = local_state.accumulated_output_offsets[i];
			idx_t len = local_state.accumulated_output_lengths[i];
			for (idx_t j = 0; j < len; j++) {
				if (local_state.accumulated_outputs[off + j] == upper64) {
					matched[i] = true;
					break;
				}
			}
		}
	}

	// ================================================================
	// Phase 6: Stash matches in this thread's local match buffer
	// ================================================================
	for (idx_t i = 0; i < N; i++) {
		if (matched[i]) {
			local_state.match_txids.push_back(std::move(local_state.accumulated_txids[i]));
			local_state.match_heights.push_back(local_state.accumulated_heights[i]);
			local_state.match_tweak_keys.push_back(std::move(local_state.accumulated_tweak_keys[i]));
		}
	}

	// Clear accumulated input after processing
	local_state.accumulated_txids.clear();
	local_state.accumulated_heights.clear();
	local_state.accumulated_tweak_keys.clear();
	local_state.accumulated_outputs.clear();
	local_state.accumulated_output_offsets.clear();
	local_state.accumulated_output_lengths.clear();
}

// ============================================================================
// Helper predicates
// ============================================================================

static bool ShouldProcessBatch(const UfsecpScanLocalState &local_state, const UfsecpScanBindData &bind_data) {
	return local_state.accumulated_txids.size() >= bind_data.batch_size;
}

// ============================================================================
// ProcessBatchGpu — GPU-accelerated BIP-352 scanning
// ============================================================================
// Full pipeline (phases 1-6 on GPU): returns match flags only.
// Fallback (phases 1-4 on GPU): returns affine (x,y) for CPU phases 5-6.

#ifdef UFSECP_GPU_ENABLED

// Build scan key GLV plan from KPlan (shared by both paths)
struct ScanGlvPlan {
	char wnaf1[130];
	char wnaf2[130];
	uint8_t k1_neg;
	uint8_t flip_phi;
	uint8_t pad0;
	uint8_t pad1;
};

static ScanGlvPlan BuildScanGlvPlan(const KPlan &plan) {
	ScanGlvPlan glv = {};
	size_t n1 = std::min(plan.wnaf1.size(), size_t(130));
	size_t n2 = std::min(plan.wnaf2.size(), size_t(130));
	for (size_t i = 0; i < n1; i++)
		glv.wnaf1[i] = static_cast<char>(plan.wnaf1[i]);
	for (size_t i = 0; i < n2; i++)
		glv.wnaf2[i] = static_cast<char>(plan.wnaf2[i]);
	glv.k1_neg = plan.neg1 ? 1 : 0;
	glv.flip_phi = (plan.neg1 != plan.neg2) ? 1 : 0;
	return glv;
}

static void ProcessBatchGpu(UfsecpScanLocalState &local_state, const UfsecpScanBindData &bind_data,
                            UfsecpScanState &global_state) {
	idx_t N = local_state.accumulated_txids.size();
	if (N == 0)
		return;

	const uint8_t *spend_xy = reinterpret_cast<const uint8_t *>(bind_data.spend_public_key_data.data());
	int num_labels = (int)bind_data.labelled_spend_keys.size();

	// Build label keys as contiguous LE bytes (consumed by both API paths below)
	std::vector<uint8_t> label_buf;
	for (auto &lsk : bind_data.labelled_spend_keys) {
		uint8_t lk[64];
		for (int i = 0; i < 4; i++) {
			uint64_t xv = lsk.x.limbs()[i], yv = lsk.y.limbs()[i];
			for (int j = 0; j < 8; j++) {
				lk[i * 8 + j] = (uint8_t)(xv >> (j * 8));
				lk[32 + i * 8 + j] = (uint8_t)(yv >> (j * 8));
			}
		}
		label_buf.insert(label_buf.end(), lk, lk + 64);
	}
	const uint8_t *label_ptr = label_buf.empty() ? nullptr : label_buf.data();

	// Marshal tweak keys into contiguous buffer (N × 64 bytes, already LE)
	std::vector<uint8_t> tweak_buf(N * 64);
	for (idx_t i = 0; i < N; i++) {
		std::memcpy(tweak_buf.data() + i * 64, local_state.accumulated_tweak_keys[i].data(), 64);
	}

	ScanGlvPlan scan_glv = BuildScanGlvPlan(bind_data.kplan);

	// ====================================================================
	// Try full pipeline (phases 1-6 on GPU)
	// ====================================================================
	if (g_gpu_launch_full) {
		// Marshal output offsets and lengths
		std::vector<uint32_t> output_offsets(N);
		std::vector<uint8_t> output_lengths(N);
		for (idx_t i = 0; i < N; i++) {
			output_offsets[i] = static_cast<uint32_t>(local_state.accumulated_output_offsets[i]);
			output_lengths[i] = static_cast<uint8_t>(local_state.accumulated_output_lengths[i]);
		}
		uint32_t total_outputs = static_cast<uint32_t>(local_state.accumulated_outputs.size());

		void *gpu_state = g_gpu_launch_full(tweak_buf.data(), local_state.accumulated_outputs.data(), total_outputs,
		                                    output_offsets.data(), output_lengths.data(), static_cast<uint32_t>(N),
		                                    local_state.assigned_gpu, &scan_glv, spend_xy, num_labels, label_ptr);

		if (gpu_state) {
			std::vector<uint8_t> match_flags(N);
			int result = g_gpu_run_full(gpu_state, match_flags.data(), static_cast<uint32_t>(N));
			g_gpu_free(gpu_state);

			if (result == 0) {
				// Stash matches in this thread's local match buffer
				for (idx_t i = 0; i < N; i++) {
					if (match_flags[i]) {
						local_state.match_txids.push_back(std::move(local_state.accumulated_txids[i]));
						local_state.match_heights.push_back(local_state.accumulated_heights[i]);
						local_state.match_tweak_keys.push_back(std::move(local_state.accumulated_tweak_keys[i]));
					}
				}

				local_state.accumulated_txids.clear();
				local_state.accumulated_heights.clear();
				local_state.accumulated_tweak_keys.clear();
				local_state.accumulated_outputs.clear();
				local_state.accumulated_output_offsets.clear();
				local_state.accumulated_output_lengths.clear();
				return;
			}
			fprintf(stderr, "[GPU] Full pipeline kernel failed (result=%d), falling back to legacy\n", result);
		} else {
			fprintf(stderr, "[GPU] Full pipeline launch returned null, falling back to legacy\n");
		}
	} else {
		fprintf(stderr, "[GPU] no full-pipeline launch fn, falling back to legacy\n");
	}

	// ====================================================================
	// Legacy path: phases 1-4 on GPU, phases 5-6 on CPU
	// ====================================================================
	const uint8_t *scan_key = reinterpret_cast<const uint8_t *>(bind_data.scan_private_key_data.data());

	void *gpu_state =
	    g_gpu_launch(scan_key, tweak_buf.data(), static_cast<uint32_t>(N), local_state.assigned_gpu, &scan_glv);

	if (!gpu_state) {
		fprintf(stderr, "[GPU] Legacy launch returned null, falling back to CPU\n");
		ProcessBatch(local_state, bind_data, global_state);
		return;
	}

	std::vector<uint8_t> out_x_bytes(N * 32);
	std::vector<uint8_t> out_y_bytes(N * 32);

	int result = g_gpu_run(gpu_state, out_x_bytes.data(), out_y_bytes.data(), static_cast<uint32_t>(N));
	g_gpu_free(gpu_state);

	if (result != 0) {
		fprintf(stderr, "[GPU] Legacy kernel failed (result=%d), falling back to CPU\n", result);
		ProcessBatch(local_state, bind_data, global_state);
		return;
	}

	// CPU Phases 5-6
	std::vector<AffinePointCompact> offsets(N);
	for (idx_t i = 0; i < N; i++) {
		std::array<uint8_t, 32> x_be, y_be;
		for (int j = 0; j < 32; j++) {
			x_be[j] = out_x_bytes[i * 32 + 31 - j];
			y_be[j] = out_y_bytes[i * 32 + 31 - j];
		}
		offsets[i].x = FieldElement::from_bytes(x_be);
		offsets[i].y = FieldElement::from_bytes(y_be);
	}

	std::vector<FieldElement> final_x(N);
	secp256k1::fast::batch_add_affine_x(bind_data.spend_x, bind_data.spend_y, offsets.data(), final_x.data(), N,
	                                    local_state.scratch);

	std::vector<bool> matched(N, false);
	for (idx_t i = 0; i < N; i++) {
		int64_t upper64 = ExtractUpper64(final_x[i]);
		idx_t off = local_state.accumulated_output_offsets[i];
		idx_t len = local_state.accumulated_output_lengths[i];
		for (idx_t j = 0; j < len; j++) {
			if (local_state.accumulated_outputs[off + j] == upper64) {
				matched[i] = true;
				break;
			}
		}
	}

	for (idx_t L = 0; L < bind_data.labelled_spend_keys.size(); L++) {
		std::vector<FieldElement> labelled_x(N);
		secp256k1::fast::batch_add_affine_x(bind_data.labelled_spend_keys[L].x, bind_data.labelled_spend_keys[L].y,
		                                    offsets.data(), labelled_x.data(), N, local_state.scratch);

		for (idx_t i = 0; i < N; i++) {
			if (matched[i])
				continue;
			int64_t upper64 = ExtractUpper64(labelled_x[i]);
			idx_t off = local_state.accumulated_output_offsets[i];
			idx_t len = local_state.accumulated_output_lengths[i];
			for (idx_t j = 0; j < len; j++) {
				if (local_state.accumulated_outputs[off + j] == upper64) {
					matched[i] = true;
					break;
				}
			}
		}
	}

	for (idx_t i = 0; i < N; i++) {
		if (matched[i]) {
			local_state.match_txids.push_back(std::move(local_state.accumulated_txids[i]));
			local_state.match_heights.push_back(local_state.accumulated_heights[i]);
			local_state.match_tweak_keys.push_back(std::move(local_state.accumulated_tweak_keys[i]));
		}
	}

	local_state.accumulated_txids.clear();
	local_state.accumulated_heights.clear();
	local_state.accumulated_tweak_keys.clear();
	local_state.accumulated_outputs.clear();
	local_state.accumulated_output_offsets.clear();
	local_state.accumulated_output_lengths.clear();
}
#endif

// ============================================================================
// Bind — validate inputs and precompute query constants
// ============================================================================

static unique_ptr<FunctionData> UfsecpScanBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	if (input.inputs.size() != 4) {
		throw InvalidInputException("ufsecp_scan requires 4 arguments: TABLE, scan_private_key BLOB, "
		                            "spend_public_key BLOB, and label_keys LIST[BLOB]");
	}

	// --- Validate scan_private_key (32-byte BLOB) ---
	auto &scalar_value = input.inputs[1];
	if (scalar_value.type().id() != LogicalTypeId::BLOB) {
		throw InvalidInputException("Second argument must be a BLOB (32-byte scan_private_key)");
	}
	string_t scan_private_key = StringValue::Get(scalar_value);
	if (scan_private_key.GetSize() != UfsecpScanBindData::SCALAR_SIZE) {
		throw InvalidInputException("scan_private_key must be exactly 32 bytes, got %llu bytes",
		                            scan_private_key.GetSize());
	}

	// --- Validate spend_public_key (64-byte BLOB) ---
	auto &spend_value = input.inputs[2];
	if (spend_value.type().id() != LogicalTypeId::BLOB) {
		throw InvalidInputException("Third argument must be a BLOB (64-byte spend_public_key)");
	}
	string_t spend_public_key = StringValue::Get(spend_value);
	if (spend_public_key.GetSize() != UfsecpScanBindData::TWEAK_KEY_SIZE) {
		throw InvalidInputException("spend_public_key must be exactly 64 bytes, got %llu bytes",
		                            spend_public_key.GetSize());
	}

	// --- Validate label_keys (LIST[BLOB], each 64 bytes) ---
	auto &label_keys_value = input.inputs[3];
	if (label_keys_value.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("Fourth argument must be a LIST[BLOB] (label keys)");
	}
	std::vector<std::string> label_keys;
	auto &list_value = ListValue::GetChildren(label_keys_value);
	for (idx_t i = 0; i < list_value.size(); i++) {
		auto &lkv = list_value[i];
		if (lkv.type().id() != LogicalTypeId::BLOB) {
			throw InvalidInputException("All elements in label_keys must be BLOBs");
		}
		string_t lk = StringValue::Get(lkv);
		if (lk.GetSize() != UfsecpScanBindData::TWEAK_KEY_SIZE) {
			throw InvalidInputException("Each label key must be exactly 64 bytes, got %llu bytes", lk.GetSize());
		}
		label_keys.push_back(std::string(lk.GetData(), lk.GetSize()));
	}

	// --- Parse optional batch_size named parameter ---
	idx_t batch_size = 300000;
	auto bs_entry = input.named_parameters.find("batch_size");
	if (bs_entry != input.named_parameters.end()) {
		auto &bsv = bs_entry->second;
		if (bsv.type().id() != LogicalTypeId::INTEGER && bsv.type().id() != LogicalTypeId::BIGINT) {
			throw InvalidInputException("batch_size parameter must be an INTEGER");
		}
		int64_t bs_int = IntegerValue::Get(bsv);
		if (bs_int <= 0) {
			throw InvalidInputException("batch_size must be positive, got %lld", bs_int);
		}
		if (bs_int > 10000000) {
			throw InvalidInputException("batch_size too large (max 10,000,000), got %lld", bs_int);
		}
		batch_size = static_cast<idx_t>(bs_int);
	}

	// --- Parse optional total_rows named parameter ---
	// When provided, used as the denominator for ufsecp_progress() so that
	// progress advances per-input-chunk (smooth) rather than per-batch
	// (granular). Caller should pass `SELECT COUNT(*) FROM <same input>` so
	// the total matches what the scan will actually consume.
	idx_t total_rows = 0;
	auto tr_entry = input.named_parameters.find("total_rows");
	if (tr_entry != input.named_parameters.end()) {
		auto &trv = tr_entry->second;
		if (trv.type().id() != LogicalTypeId::INTEGER && trv.type().id() != LogicalTypeId::BIGINT) {
			throw InvalidInputException("total_rows parameter must be an INTEGER");
		}
		int64_t tr_int = IntegerValue::Get(trv);
		if (tr_int < 0) {
			throw InvalidInputException("total_rows must be non-negative, got %lld", tr_int);
		}
		total_rows = static_cast<idx_t>(tr_int);
	}

	// --- Parse optional backend named parameter ---
	std::string backend_str = "auto";
	auto be_entry = input.named_parameters.find("backend");
	if (be_entry != input.named_parameters.end()) {
		auto &bev = be_entry->second;
		if (bev.type().id() != LogicalTypeId::VARCHAR) {
			throw InvalidInputException("backend parameter must be a string ('cpu', 'gpu', or 'auto')");
		}
		backend_str = StringValue::Get(bev);
		if (backend_str != "cpu" && backend_str != "gpu" && backend_str != "auto") {
			throw InvalidInputException("backend must be 'cpu', 'gpu', or 'auto', got '%s'", backend_str.c_str());
		}
	}

	// --- Output schema (matches cudasp_scan for Frigate compatibility) ---
	return_types.push_back(LogicalType::BLOB);    // txid
	return_types.push_back(LogicalType::INTEGER); // height
	return_types.push_back(LogicalType::BLOB);    // tweak_key
	names.push_back("txid");
	names.push_back("height");
	names.push_back("tweak_key");

	// --- Build bind data with precomputed values ---
	auto bind_data = make_uniq<UfsecpScanBindData>();
	bind_data->batch_size = batch_size;
	bind_data->total_rows = total_rows;
	bind_data->backend = backend_str;

	// Resolve backend
#ifdef UFSECP_GPU_ENABLED
	EnsureGpuDetected();
	if (bind_data->backend == "gpu") {
		if (g_gpu_backend == GpuBackend::NONE) {
			throw InvalidInputException("backend='gpu' requested but no GPU detected");
		}
		bind_data->use_gpu = true;
	} else if (bind_data->backend == "auto") {
		bind_data->use_gpu = (g_gpu_backend != GpuBackend::NONE);
	} else {
		bind_data->use_gpu = false;
	}
	// Eager backend init (e.g. build LUT) on this single thread before any
	// worker thread runs. Avoids races on lazy first-use init from workers.
	if (bind_data->use_gpu && g_gpu_ensure_ready) {
		g_gpu_ensure_ready();
	}
#else
	if (bind_data->backend == "gpu") {
		throw InvalidInputException("backend='gpu' requested but extension was compiled without GPU support");
	}
	bind_data->use_gpu = false;
#endif

	// Store raw copies
	bind_data->scan_private_key_data = std::string(scan_private_key.GetData(), scan_private_key.GetSize());
	bind_data->spend_public_key_data = std::string(spend_public_key.GetData(), spend_public_key.GetSize());
	bind_data->label_keys_data = std::move(label_keys);

	// Register progress entry for side-channel polling via ufsecp_progress().
	// If a scan with the same scan_key is already being tracked (callers may
	// run concurrent scans that share a wallet's scan key, e.g. a long
	// historical scan plus a short incremental), don't overwrite its entry —
	// the existing scan keeps owning the progress slot until it ends. This
	// scan still updates its own `progress` counters (they remain valid for
	// the streaming function) but they're not exposed via ufsecp_progress()
	// while a peer scan is in flight. The destructor's identity check ensures
	// we only erase the map entry if it still points at our ScanProgress.
	bind_data->progress = std::make_shared<ScanProgress>();
	bind_data->progress->total_rows = bind_data->total_rows;
	{
		std::lock_guard<std::mutex> lock(g_progress_mutex);
		g_progress_map.try_emplace(bind_data->scan_private_key_data, bind_data->progress);
	}

	// Precompute KPlan from scan_private_key (LE wire → Scalar → KPlan)
	const uint8_t *sk_data = reinterpret_cast<const uint8_t *>(bind_data->scan_private_key_data.data());
	Scalar scan_scalar = ScalarFromLE(sk_data);
	bind_data->kplan = KPlan::from_scalar(scan_scalar);

	// Precompute tagged hash midstate for "BIP0352/SharedSecret"
	bind_data->tag_midstate = secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");

	// Precompute spend public key affine coordinates
	const uint8_t *sp_data = reinterpret_cast<const uint8_t *>(bind_data->spend_public_key_data.data());
	bind_data->spend_x = FieldElementFromLE(sp_data);
	bind_data->spend_y = FieldElementFromLE(sp_data + 32);

	// Precompute labelled spend keys: spend_public_key + label_key[L]
	Point spend_point = Point::from_affine(bind_data->spend_x, bind_data->spend_y);
	for (auto &lk_data : bind_data->label_keys_data) {
		const uint8_t *lk_bytes = reinterpret_cast<const uint8_t *>(lk_data.data());
		FieldElement lk_x = FieldElementFromLE(lk_bytes);
		FieldElement lk_y = FieldElementFromLE(lk_bytes + 32);
		Point label_point = Point::from_affine(lk_x, lk_y);
		Point labelled = spend_point.add(label_point);

		AffinePointCompact lsk;
		lsk.x = labelled.x(); // single field inversion per label at bind time
		lsk.y = labelled.y();
		bind_data->labelled_spend_keys.push_back(lsk);
	}

	return duckdb::unique_ptr<FunctionData>(bind_data.release());
}

// ============================================================================
// Init functions
// ============================================================================

static unique_ptr<GlobalTableFunctionState> UfsecpScanInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<UfsecpScanState>();
	return duckdb::unique_ptr<GlobalTableFunctionState>(state.release());
}

static unique_ptr<LocalTableFunctionState> UfsecpScanLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                                                               GlobalTableFunctionState *global_state) {
	auto local_state = make_uniq<UfsecpScanLocalState>();

#ifdef UFSECP_GPU_ENABLED
	// Round-robin GPU assignment (same pattern as cudasp)
	auto &bind_data = input.bind_data->Cast<UfsecpScanBindData>();
	if (bind_data.use_gpu && g_num_gpus > 0) {
		static std::atomic<int> next_gpu {0};
		local_state->assigned_gpu = next_gpu.fetch_add(1) % g_num_gpus;
	}
#endif

	return duckdb::unique_ptr<LocalTableFunctionState>(local_state.release());
}

// ============================================================================
// Streaming in-out function — accumulate input, process when batch is full
// ============================================================================

static OperatorResultType UfsecpScanFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                             DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<UfsecpScanBindData>();
	auto &global_state = data_p.global_state->Cast<UfsecpScanState>();
	auto &local_state = data_p.local_state->Cast<UfsecpScanLocalState>();

	if (input.size() > 0) {
		bind_data.progress->rows_received += input.size();
		AccumulateInput(local_state, input);
		if (ShouldProcessBatch(local_state, bind_data)) {
			idx_t batch_count = local_state.accumulated_txids.size();
#ifdef UFSECP_GPU_ENABLED
			if (bind_data.use_gpu) {
				ProcessBatchGpu(local_state, bind_data, global_state);
			} else {
				ProcessBatch(local_state, bind_data, global_state);
			}
#else
			ProcessBatch(local_state, bind_data, global_state);
#endif
			bind_data.progress->rows_processed += batch_count;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================================
// Finalize — process remaining data, single-output-thread returns results
// ============================================================================

static OperatorFinalizeResultType UfsecpScanFinalFunction(ExecutionContext &context, TableFunctionInput &data_p,
                                                          DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<UfsecpScanBindData>();
	auto &state = data_p.global_state->Cast<UfsecpScanState>();
	auto &local_state = data_p.local_state->Cast<UfsecpScanLocalState>();

	// Process any remaining accumulated data from this thread (only once;
	// subsequent calls just drain match_* below).
	if (!local_state.finalized && !local_state.accumulated_txids.empty()) {
		idx_t batch_count = local_state.accumulated_txids.size();
#ifdef UFSECP_GPU_ENABLED
		if (bind_data.use_gpu) {
			ProcessBatchGpu(local_state, bind_data, state);
		} else {
			ProcessBatch(local_state, bind_data, state);
		}
#else
		ProcessBatch(local_state, bind_data, state);
#endif
		bind_data.progress->rows_processed += batch_count;
	}
	local_state.finalized = true;

	// Drain this thread's local matches in chunks of STANDARD_VECTOR_SIZE.
	// Each thread emits its own matches independently — DuckDB combines the
	// per-thread chunks into the final result. This avoids a global
	// "single output thread" claim that races with DuckDB's interleaved
	// LocalInit/Finalize lifecycle.
	idx_t total = local_state.match_txids.size();
	idx_t pos = local_state.match_position;
	if (pos < total) {
		auto &txid_result = output.data[0];
		auto &height_result = output.data[1];
		auto &tweak_key_result = output.data[2];

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - pos);

		auto txid_data = FlatVector::GetData<string_t>(txid_result);
		auto height_data = FlatVector::GetData<int32_t>(height_result);
		auto tweak_key_data = FlatVector::GetData<string_t>(tweak_key_result);

		for (idx_t i = 0; i < output_count; i++) {
			auto &txid = local_state.match_txids[pos + i];
			auto &tweak_key = local_state.match_tweak_keys[pos + i];
			txid_data[i] = StringVector::AddStringOrBlob(txid_result, string_t(txid.data(), txid.size()));
			height_data[i] = local_state.match_heights[pos + i];
			tweak_key_data[i] =
			    StringVector::AddStringOrBlob(tweak_key_result, string_t(tweak_key.data(), tweak_key.size()));
		}

		output.SetCardinality(output_count);
		local_state.match_position += output_count;

		if (local_state.match_position < total) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}
	}

	bind_data.progress->complete = true;
	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================================
// Extension registration
// ============================================================================

static void LoadInternal(ExtensionLoader &loader) {
	// Use w=12 precompute table (~5 MB) instead of default w=18 (~244 MB).
	// Smaller table avoids stack overflow during build on Windows (1 MB default thread stack)
	// and is fast enough for the CPU fallback path.
	{
		secp256k1::fast::FixedBaseConfig cfg {};
		cfg.window_bits = 12;
		cfg.thread_count = 1;
		secp256k1::fast::configure_fixed_base(cfg);
	}

	TableFunction func("ufsecp_scan",
	                   {LogicalType::TABLE, LogicalType::BLOB, LogicalType::BLOB, LogicalType::LIST(LogicalType::BLOB)},
	                   nullptr, UfsecpScanBind, UfsecpScanInit, UfsecpScanLocalInit);
	func.in_out_function = UfsecpScanFunction;
	func.in_out_function_final = UfsecpScanFinalFunction;
	func.named_parameters["batch_size"] = LogicalType::INTEGER;
	func.named_parameters["backend"] = LogicalType::VARCHAR;
	func.named_parameters["total_rows"] = LogicalType::BIGINT;
	loader.RegisterFunction(func);

	// ufsecp_set_cache_dir(path) — set precompute table cache directory
	ScalarFunction set_cache_dir_func("ufsecp_set_cache_dir", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                  [](DataChunk &args, ExpressionState &state, Vector &result) {
		                                  auto &path_vector = args.data[0];
		                                  auto path_str = path_vector.GetValue(0).ToString();
		                                  // Set the full cache file path explicitly to bypass get_default_cache_path(),
		                                  // which only returns cache_dir paths for files that already exist.
		                                  std::string cache_path = path_str + "/cache_w12.bin";
#ifdef _WIN32
		                                  _putenv_s("SECP256K1_CACHE_PATH", cache_path.c_str());
#else
		    setenv("SECP256K1_CACHE_PATH", cache_path.c_str(), 1);
#endif
		                                  // Re-configure to pick up the new env vars, then eagerly build+cache
		                                  // the tables. ensure_fixed_base_ready() is needed because the GPU scan
		                                  // path bypasses Point::generator().scalar_mul() and would never trigger
		                                  // the lazy build.
		                                  secp256k1::fast::FixedBaseConfig cfg {};
		                                  cfg.window_bits = 12;
		                                  cfg.thread_count = 1;
		                                  secp256k1::fast::configure_fixed_base(cfg);
		                                  secp256k1::fast::ensure_fixed_base_ready();
		                                  result.SetValue(0, Value(path_str));
		                                  result.SetVectorType(VectorType::CONSTANT_VECTOR);
	                                  });
	set_cache_dir_func.stability = FunctionStability::VOLATILE;
	loader.RegisterFunction(set_cache_dir_func);

	// ufsecp_backend() — returns current backend info
	ScalarFunction backend_func(
	    "ufsecp_backend", {}, LogicalType::VARCHAR, [](DataChunk &args, ExpressionState &state, Vector &result) {
		    std::string backend_str;
#ifdef UFSECP_GPU_ENABLED
		    EnsureGpuDetected();
		    if (g_gpu_backend == GpuBackend::CUDA) {
			    backend_str = "cuda (" + std::to_string(g_num_gpus) + " device" + (g_num_gpus > 1 ? "s" : "") + ")";
		    } else if (g_gpu_backend == GpuBackend::OPENCL) {
			    backend_str = "opencl (" + std::to_string(g_num_gpus) + " device" + (g_num_gpus > 1 ? "s" : "") + ")";
		    } else if (g_gpu_backend == GpuBackend::METAL) {
			    backend_str = "metal (" + std::to_string(g_num_gpus) + " device" + (g_num_gpus > 1 ? "s" : "") + ")";
		    } else {
			    backend_str = "cpu (GPU compiled, no GPU detected)";
		    }
#else
			backend_str = "cpu";
#endif
		    result.SetValue(0, Value(backend_str));
		    result.SetVectorType(VectorType::CONSTANT_VECTOR);
	    });
	backend_func.stability = FunctionStability::CONSISTENT;
	loader.RegisterFunction(backend_func);

	// ufsecp_progress(scan_key) — returns scan progress percentage (0-100), or -1 if no scan active
	ScalarFunction progress_func(
	    "ufsecp_progress", {LogicalType::BLOB}, LogicalType::DOUBLE,
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    auto &key_vector = args.data[0];
		    auto key_val = key_vector.GetValue(0);
		    string_t key = StringValue::Get(key_val);
		    std::string scan_key(key.GetData(), key.GetSize());

		    double pct = -1.0;
		    {
			    std::lock_guard<std::mutex> lock(g_progress_mutex);
			    auto it = g_progress_map.find(scan_key);
			    if (it != g_progress_map.end()) {
				    auto &sp = it->second;
				    uint64_t received = sp->rows_received.load();
				    uint64_t processed = sp->rows_processed.load();
				    // Preferred path: caller passed total_rows on
				    // ufsecp_scan, so we can report received/total
				    // which advances per-input-chunk (smooth).
				    if (sp->total_rows > 0) {
					    pct = static_cast<double>(received) / static_cast<double>(sp->total_rows) * 100.0;
					    if (pct > 100.0)
						    pct = 100.0;
				    }
				    // Fallback: processed/received (granular — jumps in
				    // batch_size increments). Used when caller didn't
				    // supply total_rows.
				    else if (received > 0) {
					    pct = static_cast<double>(processed) / static_cast<double>(received) * 100.0;
					    if (pct > 100.0)
						    pct = 100.0;
				    } else if (sp->complete) {
					    pct = 100.0;
				    } else {
					    pct = 0.0;
				    }
			    }
		    }
		    result.SetValue(0, Value::DOUBLE(pct));
		    result.SetVectorType(VectorType::CONSTANT_VECTOR);
	    });
	progress_func.stability = FunctionStability::VOLATILE;
	loader.RegisterFunction(progress_func);
}

void UfsecpExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string UfsecpExtension::Name() {
	return "ufsecp";
}

std::string UfsecpExtension::Version() const {
#ifdef EXT_VERSION_UFSECP
	return EXT_VERSION_UFSECP;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(ufsecp, loader) {
	duckdb::LoadInternal(loader);
}
}
