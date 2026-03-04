PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=ufsecp
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Forward UFSECP_ENABLE_CUDA to CMake via EXT_FLAGS
ifdef UFSECP_ENABLE_CUDA
EXT_FLAGS += -DUFSECP_ENABLE_CUDA=$(UFSECP_ENABLE_CUDA)
endif

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile
