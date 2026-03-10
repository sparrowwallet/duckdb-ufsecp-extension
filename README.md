# DuckDB UltrafastSecp256k1 Extension

A DuckDB extension for Bitcoin Silent Payments (BIP-352) scanning using [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1). Supports CPU, optional NVIDIA CUDA, and optional OpenCL GPU acceleration.

## Features

- **BIP-352 scanning**: Full Silent Payments pipeline (scalar multiply, tagged hash, generator multiply, point addition, prefix matching)
- **Label support**: Tests both base output and label-tweaked variants
- **CPU + GPU**: CPU-only by default, with optional CUDA or OpenCL GPU acceleration
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

### `ufsecp_scan(input_table, scan_private_key, spend_public_key, label_keys, [backend, batch_size])`

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

### `ufsecp_backend()`

Returns a string describing the active backend.

```sql
SELECT ufsecp_backend();
-- 'cpu', 'cuda (2 devices)', 'opencl (1 device)', or 'cpu (OpenCL compiled, no GPU detected)'
```

### `ufsecp_set_cache_dir(path)`

Sets the directory for the precomputed lookup table cache and eagerly builds the table if it doesn't exist yet. Returns the path on success.

```sql
SELECT ufsecp_set_cache_dir('/path/to/cache');
```

This writes a ~244 MB file (`cache_w18.bin`) to the specified directory. The table is used for fast fixed-base scalar multiplication and is generated once on first use.

## Precomputed table cache

UltrafastSecp256k1 generates a ~244 MB precomputed lookup table (`cache_w18.bin`) on first use for fast fixed-scalar multiplication. This file is written to the current working directory by default.

To control the cache location, call `ufsecp_set_cache_dir()` after loading the extension, or set the environment variable before loading:

```bash
export SECP256K1_CACHE_DIR=/path/to/cache/directory
```

Or set the exact file path:

```bash
export SECP256K1_CACHE_PATH=/path/to/cache_w18.bin
```

## Dependencies

- [UltrafastSecp256k1](https://github.com/shrec/UltrafastSecp256k1): High-performance secp256k1 library
- [DuckDB](https://duckdb.org/): In-process analytical database
- OpenSSL
- NVIDIA CUDA Runtime (optional, for CUDA GPU acceleration)
- OpenCL ICD loader (optional, for OpenCL GPU acceleration)

## License

MIT License — see [LICENSE](LICENSE) for details.
