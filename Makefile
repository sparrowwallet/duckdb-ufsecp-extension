PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=ufsecp
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# GPU auto-detection: enable backends based on available tools unless explicitly set.

# CUDA: auto-enable if nvcc is on PATH
ifndef UFSECP_ENABLE_CUDA
ifneq ($(shell which nvcc 2>/dev/null),)
UFSECP_ENABLE_CUDA := ON
endif
endif
ifdef UFSECP_ENABLE_CUDA
EXT_FLAGS += -DUFSECP_ENABLE_CUDA=$(UFSECP_ENABLE_CUDA)
endif

# OpenCL: auto-enable alongside CUDA (CUDA Toolkit includes OpenCL)
ifndef UFSECP_ENABLE_OPENCL
ifdef UFSECP_ENABLE_CUDA
UFSECP_ENABLE_OPENCL := ON
endif
endif
ifdef UFSECP_ENABLE_OPENCL
EXT_FLAGS += -DUFSECP_ENABLE_OPENCL=$(UFSECP_ENABLE_OPENCL)
endif

# Metal: auto-enable on macOS
ifndef UFSECP_ENABLE_METAL
ifeq ($(shell uname -s),Darwin)
UFSECP_ENABLE_METAL := ON
endif
endif
ifdef UFSECP_ENABLE_METAL
EXT_FLAGS += -DUFSECP_ENABLE_METAL=$(UFSECP_ENABLE_METAL)
endif

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile
