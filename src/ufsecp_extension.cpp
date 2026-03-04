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

#include <vector>
#include <cstring>
#include <mutex>
#include <atomic>

// Conditional CUDA support — extern "C" declarations for ufsecp_gpu.cu
#ifdef UFSECP_CUDA_ENABLED
extern "C" {
int UfsecpGpuDetect(int *num_gpus);
void *UfsecpGpuLaunchBatch(const uint8_t *scan_key, const uint8_t *tweak_data, uint32_t count, int device_id);
int UfsecpGpuRunKernels(void *state_handle, uint8_t *out_x, uint8_t *out_y, uint32_t count);
void UfsecpGpuFreeBatch(void *state_handle);
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

#ifdef UFSECP_CUDA_ENABLED
static int g_num_gpus = 0;
static bool g_gpu_detected = false;
static std::mutex g_gpu_init_mutex;

static void EnsureGpuDetected() {
	if (g_gpu_detected)
		return;
	std::lock_guard<std::mutex> lock(g_gpu_init_mutex);
	if (g_gpu_detected)
		return;
	UfsecpGpuDetect(&g_num_gpus);
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
	UfsecpScanBindData() : batch_size(300000) {
	}

	static constexpr idx_t TWEAK_KEY_SIZE = 64; // 64 bytes: uncompressed EC point (32-byte x || 32-byte y)
	static constexpr idx_t SCALAR_SIZE = 32;    // 32 bytes: scalar for EC multiplication

	idx_t batch_size;

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
};

// ============================================================================
// Local state — per-thread accumulation buffers
// ============================================================================

struct UfsecpScanLocalState : public LocalTableFunctionState {
	UfsecpScanLocalState() : finalized(false), is_output_thread(false) {
	}

	bool finalized;
	bool is_output_thread;

	// Per-thread accumulated input data (same layout as cudasp)
	vector<std::string> accumulated_txids;
	vector<int32_t> accumulated_heights;
	vector<std::string> accumulated_tweak_keys;
	vector<int64_t> accumulated_outputs;
	vector<idx_t> accumulated_output_offsets;
	vector<idx_t> accumulated_output_lengths;

	// Reusable scratch buffers (avoid per-batch heap allocation)
	std::vector<FieldElement> scratch;

#ifdef UFSECP_CUDA_ENABLED
	int assigned_gpu = -1; // GPU device ID for this thread (-1 = CPU)
#endif
};

// ============================================================================
// Global state — thread synchronization and output collection
// ============================================================================

struct UfsecpScanState : public GlobalTableFunctionState {
	UfsecpScanState() : currently_adding(0), output_position(0), output_thread_claimed(false) {
		finalize_lock = make_uniq<std::mutex>();
		output_lock = make_uniq<std::mutex>();
	}

	// Thread synchronization
	std::atomic_uint64_t currently_adding;
	unique_ptr<std::mutex> finalize_lock;

	// Global output storage — all threads write here under lock
	unique_ptr<std::mutex> output_lock;
	vector<string> output_txids;
	vector<int32_t> output_heights;
	vector<string> output_tweak_keys;
	idx_t output_position;

	// Only one thread returns output to avoid batch index conflicts
	std::atomic<bool> output_thread_claimed;
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
	// Phase 6: Write matches to global output
	// ================================================================
	global_state.output_lock->lock();
	for (idx_t i = 0; i < N; i++) {
		if (matched[i]) {
			global_state.output_txids.push_back(local_state.accumulated_txids[i]);
			global_state.output_heights.push_back(local_state.accumulated_heights[i]);
			global_state.output_tweak_keys.push_back(local_state.accumulated_tweak_keys[i]);
		}
	}
	global_state.output_lock->unlock();

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

static bool HasOutput(const UfsecpScanState &global_state) {
	return global_state.output_position < global_state.output_txids.size();
}

static bool ShouldProcessBatch(const UfsecpScanLocalState &local_state, const UfsecpScanBindData &bind_data) {
	return local_state.accumulated_txids.size() >= bind_data.batch_size;
}

// ============================================================================
// ProcessBatchGpu — GPU-accelerated BIP-352 scanning (phases 1-4 on GPU)
// ============================================================================

#ifdef UFSECP_CUDA_ENABLED
static void ProcessBatchGpu(UfsecpScanLocalState &local_state, const UfsecpScanBindData &bind_data,
                            UfsecpScanState &global_state) {
	idx_t N = local_state.accumulated_txids.size();
	if (N == 0)
		return;

	// === GPU Phases 1-4: fused kernel ===

	// Marshal tweak keys into contiguous buffer (N × 64 bytes, already LE)
	std::vector<uint8_t> tweak_buf(N * 64);
	for (idx_t i = 0; i < N; i++) {
		std::memcpy(tweak_buf.data() + i * 64, local_state.accumulated_tweak_keys[i].data(), 64);
	}

	// Launch GPU batch
	const uint8_t *scan_key = reinterpret_cast<const uint8_t *>(bind_data.scan_private_key_data.data());

	void *gpu_state =
	    UfsecpGpuLaunchBatch(scan_key, tweak_buf.data(), static_cast<uint32_t>(N), local_state.assigned_gpu);

	if (!gpu_state) {
		// GPU launch failed — fall back to CPU
		ProcessBatch(local_state, bind_data, global_state);
		return;
	}

	// Allocate host output buffers for GPU results
	std::vector<uint8_t> out_x_bytes(N * 32);
	std::vector<uint8_t> out_y_bytes(N * 32);

	int result = UfsecpGpuRunKernels(gpu_state, out_x_bytes.data(), out_y_bytes.data(), static_cast<uint32_t>(N));

	UfsecpGpuFreeBatch(gpu_state);

	if (result != 0) {
		// Kernel failed — fall back to CPU
		ProcessBatch(local_state, bind_data, global_state);
		return;
	}

	// === CPU Phases 5-6: batch affine add + match ===

	// Convert GPU output bytes to AffinePointCompact for batch_add_affine_x
	std::vector<AffinePointCompact> offsets(N);
	for (idx_t i = 0; i < N; i++) {
		// GPU output is 32 LE bytes per coordinate → FieldElement::from_bytes expects BE
		std::array<uint8_t, 32> x_be, y_be;
		for (int j = 0; j < 32; j++) {
			x_be[j] = out_x_bytes[i * 32 + 31 - j];
			y_be[j] = out_y_bytes[i * 32 + 31 - j];
		}
		offsets[i].x = FieldElement::from_bytes(x_be);
		offsets[i].y = FieldElement::from_bytes(y_be);
	}

	// Phase 5: Batch addition — base case (spend_key + output_point[i])
	std::vector<FieldElement> final_x(N);
	secp256k1::fast::batch_add_affine_x(bind_data.spend_x, bind_data.spend_y, offsets.data(), final_x.data(), N,
	                                    local_state.scratch);

	// Phase 6: Match checking — base case
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

	// Phase 6b: Label cases
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

	// Write matches to global output
	global_state.output_lock->lock();
	for (idx_t i = 0; i < N; i++) {
		if (matched[i]) {
			global_state.output_txids.push_back(local_state.accumulated_txids[i]);
			global_state.output_heights.push_back(local_state.accumulated_heights[i]);
			global_state.output_tweak_keys.push_back(local_state.accumulated_tweak_keys[i]);
		}
	}
	global_state.output_lock->unlock();

	// Clear accumulated input after processing
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
	bind_data->backend = backend_str;

	// Resolve backend
#ifdef UFSECP_CUDA_ENABLED
	EnsureGpuDetected();
	if (bind_data->backend == "gpu") {
		if (g_num_gpus == 0) {
			throw InvalidInputException("backend='gpu' requested but no CUDA GPU detected");
		}
		bind_data->use_gpu = true;
	} else if (bind_data->backend == "auto") {
		bind_data->use_gpu = (g_num_gpus > 0);
	} else {
		bind_data->use_gpu = false;
	}
#else
	if (bind_data->backend == "gpu") {
		throw InvalidInputException("backend='gpu' requested but extension was compiled without CUDA support");
	}
	bind_data->use_gpu = false;
#endif

	// Store raw copies
	bind_data->scan_private_key_data = std::string(scan_private_key.GetData(), scan_private_key.GetSize());
	bind_data->spend_public_key_data = std::string(spend_public_key.GetData(), spend_public_key.GetSize());
	bind_data->label_keys_data = std::move(label_keys);

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
	auto &state = global_state->Cast<UfsecpScanState>();
	state.currently_adding++;
	auto local_state = make_uniq<UfsecpScanLocalState>();

#ifdef UFSECP_CUDA_ENABLED
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
		AccumulateInput(local_state, input);
		if (ShouldProcessBatch(local_state, bind_data)) {
#ifdef UFSECP_CUDA_ENABLED
			if (bind_data.use_gpu) {
				ProcessBatchGpu(local_state, bind_data, global_state);
			} else {
				ProcessBatch(local_state, bind_data, global_state);
			}
#else
			ProcessBatch(local_state, bind_data, global_state);
#endif
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

	// Process any remaining accumulated data from this thread
	if (!local_state.finalized && !local_state.accumulated_txids.empty()) {
#ifdef UFSECP_CUDA_ENABLED
		if (bind_data.use_gpu) {
			ProcessBatchGpu(local_state, bind_data, state);
		} else {
			ProcessBatch(local_state, bind_data, state);
		}
#else
		ProcessBatch(local_state, bind_data, state);
#endif
	}

	// Decrement thread counter only once per thread
	if (!local_state.finalized) {
		state.finalize_lock->lock();
		state.currently_adding--;
		local_state.finalized = true;
		state.finalize_lock->unlock();
	}

	// Single-output-thread pattern: only one thread returns output
	// to avoid batch index conflicts in DuckDB
	if (!local_state.is_output_thread) {
		if (state.currently_adding != 0) {
			return OperatorFinalizeResultType::FINISHED;
		}
		bool expected = false;
		if (!state.output_thread_claimed.compare_exchange_strong(expected, true)) {
			return OperatorFinalizeResultType::FINISHED;
		}
		local_state.is_output_thread = true;
	}

	// Return output from global state in STANDARD_VECTOR_SIZE chunks
	if (HasOutput(state)) {
		auto &txid_result = output.data[0];
		auto &height_result = output.data[1];
		auto &tweak_key_result = output.data[2];

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.output_txids.size() - state.output_position);

		auto txid_data = FlatVector::GetData<string_t>(txid_result);
		auto height_data = FlatVector::GetData<int32_t>(height_result);
		auto tweak_key_data = FlatVector::GetData<string_t>(tweak_key_result);

		for (idx_t i = 0; i < output_count; i++) {
			auto &txid = state.output_txids[state.output_position + i];
			auto &tweak_key = state.output_tweak_keys[state.output_position + i];
			txid_data[i] = StringVector::AddStringOrBlob(txid_result, string_t(txid.data(), txid.size()));
			height_data[i] = state.output_heights[state.output_position + i];
			tweak_key_data[i] =
			    StringVector::AddStringOrBlob(tweak_key_result, string_t(tweak_key.data(), tweak_key.size()));
		}

		output.SetCardinality(output_count);
		state.output_position += output_count;

		if (HasOutput(state)) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================================
// Extension registration
// ============================================================================

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction func("ufsecp_scan",
	                   {LogicalType::TABLE, LogicalType::BLOB, LogicalType::BLOB, LogicalType::LIST(LogicalType::BLOB)},
	                   nullptr, UfsecpScanBind, UfsecpScanInit, UfsecpScanLocalInit);
	func.in_out_function = UfsecpScanFunction;
	func.in_out_function_final = UfsecpScanFinalFunction;
	func.named_parameters["batch_size"] = LogicalType::INTEGER;
	func.named_parameters["backend"] = LogicalType::VARCHAR;
	loader.RegisterFunction(func);

	// ufsecp_backend() — returns current backend info
	ScalarFunction backend_func(
	    "ufsecp_backend", {}, LogicalType::VARCHAR, [](DataChunk &args, ExpressionState &state, Vector &result) {
		    std::string backend_str;
#ifdef UFSECP_CUDA_ENABLED
		    EnsureGpuDetected();
		    if (g_num_gpus > 0) {
			    backend_str = "gpu (" + std::to_string(g_num_gpus) + " device" + (g_num_gpus > 1 ? "s" : "") + ")";
		    } else {
			    backend_str = "cpu (CUDA compiled, no GPU detected)";
		    }
#else
			backend_str = "cpu";
#endif
		    result.SetValue(0, Value(backend_str));
		    result.SetVectorType(VectorType::CONSTANT_VECTOR);
	    });
	backend_func.stability = FunctionStability::CONSISTENT;
	loader.RegisterFunction(backend_func);
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
