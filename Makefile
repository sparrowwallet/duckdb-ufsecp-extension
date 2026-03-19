PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=ufsecp
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Forward GPU toggles to CMake via EXT_FLAGS (can be combined)
ifdef UFSECP_ENABLE_CUDA
EXT_FLAGS += -DUFSECP_ENABLE_CUDA=$(UFSECP_ENABLE_CUDA)
endif
ifdef UFSECP_ENABLE_OPENCL
EXT_FLAGS += -DUFSECP_ENABLE_OPENCL=$(UFSECP_ENABLE_OPENCL)
endif

# Auto-enable Metal on macOS (both arm64 and amd64) unless explicitly set.
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
