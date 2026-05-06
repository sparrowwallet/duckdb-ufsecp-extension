# DuckDB UltrafastSecp256k1 Extension

A DuckDB extension for Bitcoin Silent Payments (BIP-352) scanning using [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1). Supports CPU, optional NVIDIA CUDA, optional OpenCL, and optional Apple Metal GPU acceleration.

## Features

- **BIP-352 scanning**: Full Silent Payments pipeline (scalar multiply, tagged hash, generator multiply, point addition, prefix matching)
- **Label support**: Tests both base output and label-tweaked variants
- **CPU + GPU**: CPU-only by default, with optional CUDA, OpenCL, or Metal GPU acceleration
- **Batch processing**: Configurable batch sizes for optimal throughput
- **Automatic backend selection**: Detects available GPUs and selects the best backend

## Building

### Prerequisites

- CMake 3.18+
- C++ compiler with C++20 support
- OpenSSL
- Git

For CUDA GPU support (optional):
- NVIDIA GPU with compute capability 8.0+ (Ampere, Ada Lovelace, Hopper, Blackwell)
- CUDA Toolkit 12.8+

For OpenCL GPU support (optional):
- Any GPU with OpenCL 1.2+ support (NVIDIA, AMD, Intel)
- OpenCL ICD loader and GPU driver

For Metal GPU support (optional, macOS only):
- Apple Silicon Mac (M1 or later)
- macOS with Metal framework (included with Xcode)

### Build steps

```bash
git clone --recursive https://github.com/sparrowwallet/duckdb-ufsecp-extension.git
cd duckdb-ufsecp-extension
```

CPU-only build:
```bash
GEN=ninja make
```

With CUDA GPU support:
```bash
UFSECP_ENABLE_CUDA=ON GEN=ninja make
```

With OpenCL GPU support:
```bash
UFSECP_ENABLE_OPENCL=ON GEN=ninja make
```

With both CUDA and OpenCL (runtime auto-selection: CUDA preferred → OpenCL fallback → CPU):
```bash
UFSECP_ENABLE_CUDA=ON UFSECP_ENABLE_OPENCL=ON GEN=ninja make
```

With Metal GPU support (macOS Apple Silicon):
```bash
UFSECP_ENABLE_METAL=ON GEN=ninja make
```

Run tests:
```bash
GEN=ninja make test
```

The compiled extension is at `build/release/extension/ufsecp/ufsecp.duckdb_extension`.
The compiled DuckDB binary at `build/release/duckdb` loads the extension automatically.

### Loading the extension

```sql
LOAD 'path/to/ufsecp.duckdb_extension';
```

## Functions

### `ufsecp_scan(input_table, scan_private_key, spend_public_key, label_keys, [backend, batch_size, total_rows])`

Scans a table of transactions for BIP-352 Silent Payments matches.

**Parameters:**
- `input_table` (TABLE): Input table with columns:
  - `txid` (BLOB): 32-byte transaction ID
  - `height` (INTEGER): Block height
  - `tweak_key` (BLOB): 64-byte uncompressed EC point (32-byte x || 32-byte y, little-endian)
  - `outputs` (BIGINT[]): Array of output prefix values (first 8 bytes of x-coordinates as big-endian integers)
- `scan_private_key` (BLOB): 32-byte scan private key (little-endian)
- `spend_public_key` (BLOB): 64-byte uncompressed spend public key (32-byte x || 32-byte y, little-endian)
- `label_keys` (LIST[BLOB]): Array of 64-byte uncompressed label public keys (can be empty)
- `backend` (VARCHAR, optional): `'cpu'`, `'gpu'`, or `'auto'` (default: `'auto'`)
- `batch_size` (INTEGER, optional): Rows per processing batch (default: 300000)
- `total_rows` (BIGINT, optional): Expected number of input rows the scan will consume. When provided, `ufsecp_progress` reports `received / total_rows` for smooth per-chunk progress. When omitted (or 0), progress falls back to `processed / received`, which advances in `batch_size` increments. Pass `SELECT COUNT(*) FROM <same input>` (with the same `WHERE` filter the scan uses) to keep numerator and denominator aligned.

**Returns:** TABLE with columns:
- `txid` (BLOB): Transaction ID of matching transaction
- `height` (INTEGER): Block height of matching transaction
- `tweak_key` (BLOB): Tweak key that produced the match

**Example:**
```sql
SELECT hex(txid), height
FROM ufsecp_scan(
    (SELECT txid, height, tweak_key, outputs FROM tweak),
    from_hex('0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c'),
    from_hex('36cf8fcd4d4890ab6c1083aeb5b50c260c20acda7839120e3575836f6d85c95ce0d705e31ff9fdcce67a8f3598871c6dfbe6bcde8a51cb7b48b0f95be0ea94de'),
    [from_hex('cd63f9212a2deebde8a71e9ea23f6f958c47c41d2ed74b9617fe6fb554d1524e292fabddbdcbb643eafc328875c46d75a1d697b2b31c42d38aa93f85eab34bc1')]
);
```

**Example with smooth `ufsecp_progress` reporting:** pass `total_rows` (the row count for the same input subquery) so that polling `ufsecp_progress(scan_key)` from another connection advances per-chunk instead of per-batch.

```sql
-- Compute the total once (cheap on tweak: row-group zonemaps make this ~1 ms).
SELECT COUNT(*) FROM tweak WHERE height >= 800000;
-- → say 73667836

-- Pass it into the scan.
SELECT hex(txid), height
FROM ufsecp_scan(
    (SELECT txid, height, tweak_key, outputs FROM tweak WHERE height >= 800000),
    from_hex('...'), from_hex('...'), [from_hex('...')],
    total_rows := 73667836
);
```

### `ufsecp_backend()`

Returns a string describing the active backend.

```sql
SELECT ufsecp_backend();
-- 'cpu', 'cuda (2 devices)', 'opencl (1 device)', 'metal (1 device)', or 'cpu (OpenCL compiled, no GPU detected)'
```

### `ufsecp_set_cache_dir(path)`

Sets the directory for the precomputed lookup table cache and eagerly builds the table if it doesn't exist yet. Returns the path on success.

```sql
SELECT ufsecp_set_cache_dir('/path/to/cache');
```

This writes a ~6 MB file (`cache_w12.bin`) to the specified directory. The table is used for fast fixed-base scalar multiplication and is generated once on first use.

### `ufsecp_progress(scan_key)`

Returns the progress of an active scan as a percentage (0-100), or -1 if no scan is in progress for the given key. Used as a side-channel for progress reporting since DuckDB's `QueryProgress` does not track `in_out_function` table functions.

**Parameters:**
- `scan_key` (BLOB): 32-byte scan private key (same key passed to `ufsecp_scan`)

**Returns:** DOUBLE
- `-1.0`: No scan in progress for this key
- `0.0 - 100.0`: Scan progress as a percentage

```sql
SELECT ufsecp_progress(from_hex('0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c'));
```

The percentage is computed in one of two ways depending on whether the caller supplied `total_rows` to `ufsecp_scan`:

- **Smooth mode (recommended):** `received / total_rows × 100`. Each input chunk (~2048 rows) advances progress by a fraction of a percent — values move continuously. Capped at 100 if `total_rows` underestimates the actual input.
- **Fallback mode (no `total_rows`):** `processed / received × 100`. Updates only when a batch (`batch_size` rows by default) finishes processing, so progress jumps in coarse steps of roughly `batch_size / total_input` and may plateau between batches.

For Frigate-style usage where a `WHERE height >= ?` filter is applied to the input subquery, run a `SELECT COUNT(*)` with the same filter once just before the scan and pass the result as `total_rows`. The COUNT is essentially free against `tweak` thanks to row-group zonemaps on `height` (~1–2 ms warm, ≤20 ms cold) — far cheaper than the scan it gates.

Progress is tracked per scan key. The entry is created when `ufsecp_scan` binds and removed when the query completes or is cancelled.

## Precomputed table cache

UltrafastSecp256k1 generates a ~6 MB precomputed lookup table (`cache_w12.bin`) on first use for fast fixed-base scalar multiplication. This file is written to the current working directory by default.

To control the cache location, call `ufsecp_set_cache_dir()` after loading the extension, or set the environment variable before loading:

```bash
export SECP256K1_CACHE_DIR=/path/to/cache/directory
```

Or set the exact file path:

```bash
export SECP256K1_CACHE_PATH=/path/to/cache_w12.bin
```

## Dependencies

- [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1): High-performance secp256k1 library
- [DuckDB](https://duckdb.org/): In-process analytical database
- OpenSSL
- NVIDIA CUDA Runtime (optional, for CUDA GPU acceleration)
- OpenCL ICD loader (optional, for OpenCL GPU acceleration)
- Apple Metal framework (optional, for Metal GPU acceleration on macOS)

## License

MIT License — see [LICENSE](LICENSE) for details.
