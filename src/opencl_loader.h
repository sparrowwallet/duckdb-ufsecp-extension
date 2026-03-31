// opencl_loader.h — runtime OpenCL loader
// Call opencl_loader_init() before any cl* function. Returns true if available.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

bool opencl_loader_init(void);

#ifdef __cplusplus
}
#endif
