cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME ufsecp)

find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

# --- CUDA toggle (default OFF) ---
option(UFSECP_ENABLE_CUDA "Build with CUDA GPU acceleration" OFF)

if(UFSECP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Set GPU architectures (Ampere+)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 120)
        message(STATUS "ufsecp: CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(WARNING "UFSECP_ENABLE_CUDA=ON but no CUDA compiler found. Building CPU-only.")
        set(UFSECP_ENABLE_CUDA OFF)
    endif()
endif()

project(${TARGET_NAME})

# UltrafastSecp256k1 requires C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build UltrafastSecp256k1 CPU library (disable tests/bench/examples/CABI/install)
set(SECP256K1_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_CABI OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_METAL OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_OPENCL OFF CACHE BOOL "" FORCE)
set(SECP256K1_INSTALL OFF CACHE BOOL "" FORCE)

# We do NOT build secp256k1_cuda_lib — all CUDA device functions are
# header-only (__device__ inline) in the .cuh headers. Our ufsecp_gpu.cu
# includes them directly, matching the cudasp/gECC header-only pattern.
# This avoids CUDA separable compilation and the nvlink device-link step
# that would fail on the DuckDB static library.
set(SECP256K1_BUILD_CUDA OFF CACHE BOOL "" FORCE)

add_subdirectory(UltrafastSecp256k1)

include_directories(src/include)
include_directories(UltrafastSecp256k1/cpu/include)

# --- Source files ---
# When CUDA is enabled, all sources are compiled as CUDA (same approach as
# cudasp extension) so nvcc handles the entire target uniformly.
set(EXTENSION_SOURCES src/ufsecp_extension.cpp)
if(UFSECP_ENABLE_CUDA)
    list(APPEND EXTENSION_SOURCES src/ufsecp_gpu.cu)
    include_directories(UltrafastSecp256k1/cuda/include)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# --- Link libraries ---
set(UFSECP_LIBS OpenSSL::SSL OpenSSL::Crypto fastsecp256k1)
if(UFSECP_ENABLE_CUDA)
    list(APPEND UFSECP_LIBS CUDA::cudart_static)
endif()

target_link_libraries(${EXTENSION_NAME} ${UFSECP_LIBS})
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${UFSECP_LIBS})

# --- Compile definitions and options ---
if(UFSECP_ENABLE_CUDA)
    target_compile_definitions(${EXTENSION_NAME} PRIVATE UFSECP_CUDA_ENABLED)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE UFSECP_CUDA_ENABLED)

    # Compile .cpp as CUDA so the entire target is CUDA-aware
    set_source_files_properties(src/ufsecp_extension.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(src/ufsecp_gpu.cu PROPERTIES LANGUAGE CUDA)
    target_compile_options(${EXTENSION_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >)
    target_compile_options(${LOADABLE_EXTENSION_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >)
endif()

install(
  TARGETS ${EXTENSION_NAME} fastsecp256k1
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
