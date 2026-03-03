cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME ufsecp)

find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# UltrafastSecp256k1 requires C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build UltrafastSecp256k1 CPU library (disable tests/bench/examples/CABI/install)
set(SECP256K1_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_CABI OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_CUDA OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_METAL OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_OPENCL OFF CACHE BOOL "" FORCE)
set(SECP256K1_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(UltrafastSecp256k1)

include_directories(src/include)
include_directories(UltrafastSecp256k1/cpu/include)

set(EXTENSION_SOURCES src/ufsecp_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto fastsecp256k1)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto fastsecp256k1)

install(
  TARGETS ${EXTENSION_NAME} fastsecp256k1
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
