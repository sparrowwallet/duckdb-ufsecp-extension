cmake_minimum_required(VERSION 3.18)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0 CACHE STRING "Minimum macOS deployment version" FORCE)
endif()

set(TARGET_NAME ufsecp)

find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

# --- GPU toggles (default OFF, can be combined) ---
option(UFSECP_ENABLE_CUDA "Build with CUDA GPU acceleration" OFF)
option(UFSECP_ENABLE_OPENCL "Build with OpenCL GPU acceleration" OFF)
option(UFSECP_ENABLE_METAL "Build with Metal GPU acceleration (macOS)" OFF)

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

if(UFSECP_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "ufsecp: OpenCL enabled (${OpenCL_LIBRARY})")
    else()
        message(WARNING "UFSECP_ENABLE_OPENCL=ON but OpenCL not found. Building CPU-only.")
        set(UFSECP_ENABLE_OPENCL OFF)
    endif()
endif()

if(UFSECP_ENABLE_METAL)
    if(APPLE)
        enable_language(OBJCXX)
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            message(STATUS "ufsecp: Metal enabled")
        else()
            message(WARNING "UFSECP_ENABLE_METAL=ON but Metal framework not found. Building CPU-only.")
            set(UFSECP_ENABLE_METAL OFF)
        endif()
    else()
        message(WARNING "UFSECP_ENABLE_METAL=ON on non-Apple platform. Building CPU-only.")
        set(UFSECP_ENABLE_METAL OFF)
    endif()
endif()

project(${TARGET_NAME})

# UltrafastSecp256k1 requires C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build UltrafastSecp256k1 CPU library (disable tests/bench/examples/CABI/install)
set(SECP256K1_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_CABI OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_METAL OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_ROCM OFF CACHE BOOL "" FORCE)
set(SECP256K1_INSTALL OFF CACHE BOOL "" FORCE)

# Default to portable x86-64-v3 (AVX2/FMA/BMI2) on x86_64 unless overridden.
# Override with -DSECP256K1_MARCH=native for local benchmarking.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64")
    if(NOT DEFINED SECP256K1_MARCH OR SECP256K1_MARCH STREQUAL "")
        set(SECP256K1_MARCH "x86-64-v2" CACHE STRING "" FORCE)
    endif()
endif()

# We do NOT build secp256k1_cuda_lib — all CUDA device functions are
# header-only (__device__ inline) in the .cuh headers. Our ufsecp_gpu.cu
# includes them directly, matching the cudasp/gECC header-only pattern.
# This avoids CUDA separable compilation and the nvlink device-link step
# that would fail on the DuckDB static library.
set(SECP256K1_BUILD_CUDA OFF CACHE BOOL "" FORCE)

# Enable UltrafastSecp256k1's OpenCL library when OpenCL is requested.
# Unlike CUDA, the OpenCL backend is a standard C++ static library —
# kernels are embedded as strings and compiled at runtime by the GPU driver.
if(UFSECP_ENABLE_OPENCL)
    set(SECP256K1_BUILD_OPENCL ON CACHE BOOL "" FORCE)
else()
    set(SECP256K1_BUILD_OPENCL OFF CACHE BOOL "" FORCE)
endif()

if(UFSECP_ENABLE_METAL)
    set(SECP256K1_BUILD_METAL ON CACHE BOOL "" FORCE)
else()
    set(SECP256K1_BUILD_METAL OFF CACHE BOOL "" FORCE)
endif()

set(UFSECP_REFRESH_SOURCE_GRAPH OFF CACHE BOOL "" FORCE)

add_subdirectory(UltrafastSecp256k1)

include_directories(src/include)
include_directories(UltrafastSecp256k1/cpu/include)

# --- Source files ---
set(EXTENSION_SOURCES src/ufsecp_extension.cpp)
if(UFSECP_ENABLE_CUDA)
    list(APPEND EXTENSION_SOURCES src/ufsecp_gpu.cu)
    include_directories(UltrafastSecp256k1/cuda/include)
endif()
if(UFSECP_ENABLE_OPENCL)
    list(APPEND EXTENSION_SOURCES src/ufsecp_gpu_opencl.cpp)
    include_directories(UltrafastSecp256k1/opencl/include)
    include_directories(UltrafastSecp256k1/include)
endif()
if(UFSECP_ENABLE_METAL)
    list(APPEND EXTENSION_SOURCES src/ufsecp_gpu_metal.mm)
    include_directories(UltrafastSecp256k1/metal/include)
    include_directories(UltrafastSecp256k1/include)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# --- Link libraries ---
set(UFSECP_LIBS OpenSSL::SSL OpenSSL::Crypto fastsecp256k1)
if(UFSECP_ENABLE_CUDA)
    list(APPEND UFSECP_LIBS CUDA::cudart_static)
endif()
if(UFSECP_ENABLE_OPENCL)
    list(APPEND UFSECP_LIBS secp256k1_opencl ${OpenCL_LIBRARY})
endif()
if(UFSECP_ENABLE_METAL)
    list(APPEND UFSECP_LIBS secp256k1_metal_lib ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

target_link_libraries(${EXTENSION_NAME} ${UFSECP_LIBS})
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${UFSECP_LIBS})

# --- Compile definitions and options ---
if(UFSECP_ENABLE_CUDA)
    target_compile_definitions(${EXTENSION_NAME} PRIVATE UFSECP_CUDA_ENABLED UFSECP_GPU_ENABLED)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE UFSECP_CUDA_ENABLED UFSECP_GPU_ENABLED)

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

if(UFSECP_ENABLE_OPENCL)
    # Generate embedded fused kernel source header
    set(OPENCL_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/UltrafastSecp256k1/opencl/kernels)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opencl_fused_kernel_source.h
        COMMAND ${CMAKE_COMMAND}
            -DKERNEL_DIR=${OPENCL_KERNEL_DIR}
            -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/opencl_fused_kernel_source.h
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_opencl_fused_kernel.cmake
        DEPENDS
            ${OPENCL_KERNEL_DIR}/secp256k1_field.cl
            ${OPENCL_KERNEL_DIR}/secp256k1_point.cl
            ${OPENCL_KERNEL_DIR}/secp256k1_gen_table_w8.cl
            ${OPENCL_KERNEL_DIR}/secp256k1_extended.cl
            ${OPENCL_KERNEL_DIR}/secp256k1_affine.cl
            ${OPENCL_KERNEL_DIR}/secp256k1_bip352.cl
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_opencl_fused_kernel.cmake
        COMMENT "Embedding OpenCL fused BIP-352 kernel source"
    )
    add_custom_target(opencl_fused_kernel_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/opencl_fused_kernel_source.h)
    add_dependencies(${EXTENSION_NAME} opencl_fused_kernel_header)
    add_dependencies(${LOADABLE_EXTENSION_NAME} opencl_fused_kernel_header)
    target_include_directories(${EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_compile_definitions(${EXTENSION_NAME} PRIVATE UFSECP_OPENCL_ENABLED UFSECP_GPU_ENABLED)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE UFSECP_OPENCL_ENABLED UFSECP_GPU_ENABLED)
endif()

if(UFSECP_ENABLE_METAL)
    # Generate embedded shader source header
    set(METAL_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/UltrafastSecp256k1/metal/shaders)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/metal_shader_source.h
        COMMAND ${CMAKE_COMMAND}
            -DSHADER_DIR=${METAL_SHADER_DIR}
            -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/metal_shader_source.h
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal_shaders.cmake
        DEPENDS
            ${METAL_SHADER_DIR}/secp256k1_field.h
            ${METAL_SHADER_DIR}/secp256k1_point.h
            ${METAL_SHADER_DIR}/secp256k1_bloom.h
            ${METAL_SHADER_DIR}/secp256k1_extended.h
            ${METAL_SHADER_DIR}/secp256k1_hash160.h
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_metal_shaders.cmake
        COMMENT "Embedding Metal shader source into header"
    )
    add_custom_target(metal_shader_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/metal_shader_source.h)
    add_dependencies(${EXTENSION_NAME} metal_shader_header)
    add_dependencies(${LOADABLE_EXTENSION_NAME} metal_shader_header)
    target_include_directories(${EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_compile_definitions(${EXTENSION_NAME} PRIVATE UFSECP_METAL_ENABLED UFSECP_GPU_ENABLED)
    target_compile_definitions(${LOADABLE_EXTENSION_NAME} PRIVATE UFSECP_METAL_ENABLED UFSECP_GPU_ENABLED)
    target_compile_options(${EXTENSION_NAME} PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
    target_compile_options(${LOADABLE_EXTENSION_NAME} PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
endif()

set(INSTALL_TARGETS ${EXTENSION_NAME} fastsecp256k1)
if(UFSECP_ENABLE_OPENCL)
    list(APPEND INSTALL_TARGETS secp256k1_opencl)
endif()
if(UFSECP_ENABLE_METAL)
    list(APPEND INSTALL_TARGETS secp256k1_metal_lib)
endif()

install(
  TARGETS ${INSTALL_TARGETS}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
