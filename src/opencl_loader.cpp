// opencl_loader.cpp — runtime OpenCL loader via dlopen/dlsym
//
// Provides cl* symbols that the secp256k1_opencl static library (and our own
// OpenCL code) link against. At runtime, opencl_loader_init() loads the real
// libOpenCL.so via dlopen and resolves all symbols. If libOpenCL is not
// installed, the extension still loads — OpenCL is simply unavailable.

#ifdef UFSECP_OPENCL_ENABLED

#include <CL/cl.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Real function pointers, loaded at runtime
static cl_int (*r_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *) = nullptr;
static cl_int (*r_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *) = nullptr;
static cl_int (*r_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *) = nullptr;
static cl_context (*r_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                       void(CL_CALLBACK *)(const char *, const void *, size_t, void *), void *,
                                       cl_int *) = nullptr;
static cl_int (*r_clReleaseContext)(cl_context) = nullptr;
static cl_command_queue (*r_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties,
                                                  cl_int *) = nullptr;
static cl_int (*r_clReleaseCommandQueue)(cl_command_queue) = nullptr;
static cl_int (*r_clFinish)(cl_command_queue) = nullptr;
static cl_mem (*r_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *) = nullptr;
static cl_int (*r_clReleaseMemObject)(cl_mem) = nullptr;
static cl_int (*r_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint,
                                       const cl_event *, cl_event *) = nullptr;
static cl_int (*r_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint,
                                        const cl_event *, cl_event *) = nullptr;
static cl_program (*r_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *,
                                                 cl_int *) = nullptr;
static cl_int (*r_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *,
                                  void(CL_CALLBACK *)(cl_program, void *), void *) = nullptr;
static cl_int (*r_clReleaseProgram)(cl_program) = nullptr;
static cl_int (*r_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *,
                                         size_t *) = nullptr;
static cl_kernel (*r_clCreateKernel)(cl_program, const char *, cl_int *) = nullptr;
static cl_int (*r_clReleaseKernel)(cl_kernel) = nullptr;
static cl_int (*r_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *) = nullptr;
static cl_int (*r_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
                                          const size_t *, cl_uint, const cl_event *, cl_event *) = nullptr;
static cl_int (*r_clGetCommandQueueInfo)(cl_command_queue, cl_command_queue_info, size_t, void *, size_t *) = nullptr;
static cl_int (*r_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *) = nullptr;
static cl_int (*r_clFlush)(cl_command_queue) = nullptr;
static cl_int (*r_clEnqueueFillBuffer)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint,
                                       const cl_event *, cl_event *) = nullptr;

static bool g_opencl_loaded = false;
static bool g_opencl_attempted = false;

#ifdef _WIN32
#define LOAD_SYM(handle, name) (decltype(r_##name)) GetProcAddress((HMODULE)(handle), #name)
#else
#define LOAD_SYM(handle, name) (decltype(r_##name)) dlsym(handle, #name)
#endif

extern "C" bool opencl_loader_init(void) {
	if (g_opencl_attempted)
		return g_opencl_loaded;
	g_opencl_attempted = true;

#ifdef _WIN32
	void *handle = (void *)LoadLibraryA("OpenCL.dll");
#elif defined(__APPLE__)
	void *handle = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY);
#else
	void *handle = dlopen("libOpenCL.so.1", RTLD_LAZY);
	if (!handle)
		handle = dlopen("libOpenCL.so", RTLD_LAZY);
#endif

	if (!handle)
		return false;

	r_clGetPlatformIDs = LOAD_SYM(handle, clGetPlatformIDs);
	r_clGetDeviceIDs = LOAD_SYM(handle, clGetDeviceIDs);
	r_clGetDeviceInfo = LOAD_SYM(handle, clGetDeviceInfo);
	r_clCreateContext = LOAD_SYM(handle, clCreateContext);
	r_clReleaseContext = LOAD_SYM(handle, clReleaseContext);
	r_clCreateCommandQueue = LOAD_SYM(handle, clCreateCommandQueue);
	r_clReleaseCommandQueue = LOAD_SYM(handle, clReleaseCommandQueue);
	r_clFinish = LOAD_SYM(handle, clFinish);
	r_clCreateBuffer = LOAD_SYM(handle, clCreateBuffer);
	r_clReleaseMemObject = LOAD_SYM(handle, clReleaseMemObject);
	r_clEnqueueReadBuffer = LOAD_SYM(handle, clEnqueueReadBuffer);
	r_clEnqueueWriteBuffer = LOAD_SYM(handle, clEnqueueWriteBuffer);
	r_clCreateProgramWithSource = LOAD_SYM(handle, clCreateProgramWithSource);
	r_clBuildProgram = LOAD_SYM(handle, clBuildProgram);
	r_clReleaseProgram = LOAD_SYM(handle, clReleaseProgram);
	r_clGetProgramBuildInfo = LOAD_SYM(handle, clGetProgramBuildInfo);
	r_clCreateKernel = LOAD_SYM(handle, clCreateKernel);
	r_clReleaseKernel = LOAD_SYM(handle, clReleaseKernel);
	r_clSetKernelArg = LOAD_SYM(handle, clSetKernelArg);
	r_clEnqueueNDRangeKernel = LOAD_SYM(handle, clEnqueueNDRangeKernel);
	r_clGetCommandQueueInfo = LOAD_SYM(handle, clGetCommandQueueInfo);
	r_clGetPlatformInfo = LOAD_SYM(handle, clGetPlatformInfo);
	r_clFlush = LOAD_SYM(handle, clFlush);
	r_clEnqueueFillBuffer = LOAD_SYM(handle, clEnqueueFillBuffer);

	g_opencl_loaded = (r_clGetPlatformIDs && r_clCreateContext && r_clCreateCommandQueue && r_clCreateBuffer &&
	                   r_clSetKernelArg && r_clEnqueueNDRangeKernel);

	return g_opencl_loaded;
}

// ============================================================================
// Trampoline functions — provide cl* symbols for secp256k1_opencl and our code
// ============================================================================

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint n, cl_platform_id *p, cl_uint *np) CL_API_SUFFIX__VERSION_1_0 {
	return r_clGetPlatformIDs ? r_clGetPlatformIDs(n, p, np) : CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                                                  size_t param_value_size, void *param_value,
                                                  size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
	return r_clGetPlatformInfo
	           ? r_clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret)
	           : CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries,
                                               cl_device_id *devices, cl_uint *num_devices) CL_API_SUFFIX__VERSION_1_0 {
	return r_clGetDeviceIDs ? r_clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices)
	                        : CL_DEVICE_NOT_FOUND;
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size,
                                                void *param_value,
                                                size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
	return r_clGetDeviceInfo
	           ? r_clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret)
	           : CL_INVALID_DEVICE;
}

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(const cl_context_properties *properties, cl_uint num_devices,
                                                    const cl_device_id *devices,
                                                    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t,
                                                                                  void *),
                                                    void *user_data, cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
	if (r_clCreateContext)
		return r_clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
	if (errcode_ret)
		*errcode_ret = CL_INVALID_PLATFORM;
	return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0 {
	return r_clReleaseContext ? r_clReleaseContext(context) : CL_INVALID_CONTEXT;
}

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context context, cl_device_id device,
                                                               cl_command_queue_properties properties,
                                                               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
	if (r_clCreateCommandQueue)
		return r_clCreateCommandQueue(context, device, properties, errcode_ret);
	if (errcode_ret)
		*errcode_ret = CL_INVALID_CONTEXT;
	return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
	return r_clReleaseCommandQueue ? r_clReleaseCommandQueue(command_queue) : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
	return r_clFinish ? r_clFinish(command_queue) : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr,
                                               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
	if (r_clCreateBuffer)
		return r_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
	if (errcode_ret)
		*errcode_ret = CL_INVALID_CONTEXT;
	return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0 {
	return r_clReleaseMemObject ? r_clReleaseMemObject(memobj) : CL_INVALID_MEM_OBJECT;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer,
                                                    cl_bool blocking_read, size_t offset, size_t size, void *ptr,
                                                    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                                    cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
	return r_clEnqueueReadBuffer ? r_clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr,
	                                                     num_events_in_wait_list, event_wait_list, event)
	                             : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                                                     cl_bool blocking_write, size_t offset, size_t size,
                                                     const void *ptr, cl_uint num_events_in_wait_list,
                                                     const cl_event *event_wait_list,
                                                     cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
	return r_clEnqueueWriteBuffer ? r_clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size, ptr,
	                                                       num_events_in_wait_list, event_wait_list, event)
	                              : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings,
                                                              const size_t *lengths,
                                                              cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
	if (r_clCreateProgramWithSource)
		return r_clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
	if (errcode_ret)
		*errcode_ret = CL_INVALID_CONTEXT;
	return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list,
                                               const char *options, void(CL_CALLBACK *pfn_notify)(cl_program, void *),
                                               void *user_data) CL_API_SUFFIX__VERSION_1_0 {
	return r_clBuildProgram ? r_clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data)
	                        : CL_INVALID_PROGRAM;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0 {
	return r_clReleaseProgram ? r_clReleaseProgram(program) : CL_INVALID_PROGRAM;
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(cl_program program, cl_device_id device,
                                                      cl_program_build_info param_name, size_t param_value_size,
                                                      void *param_value,
                                                      size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
	return r_clGetProgramBuildInfo ? r_clGetProgramBuildInfo(program, device, param_name, param_value_size, param_value,
	                                                         param_value_size_ret)
	                               : CL_INVALID_PROGRAM;
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program program, const char *kernel_name,
                                                  cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
	if (r_clCreateKernel)
		return r_clCreateKernel(program, kernel_name, errcode_ret);
	if (errcode_ret)
		*errcode_ret = CL_INVALID_PROGRAM;
	return nullptr;
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0 {
	return r_clReleaseKernel ? r_clReleaseKernel(kernel) : CL_INVALID_KERNEL;
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                                               const void *arg_value) CL_API_SUFFIX__VERSION_1_0 {
	return r_clSetKernelArg ? r_clSetKernelArg(kernel, arg_index, arg_size, arg_value) : CL_INVALID_KERNEL;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel,
                                                       cl_uint work_dim, const size_t *global_work_offset,
                                                       const size_t *global_work_size, const size_t *local_work_size,
                                                       cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                                       cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
	return r_clEnqueueNDRangeKernel
	           ? r_clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size,
	                                      local_work_size, num_events_in_wait_list, event_wait_list, event)
	           : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name,
                                                      size_t param_value_size, void *param_value,
                                                      size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
	return r_clGetCommandQueueInfo
	           ? r_clGetCommandQueueInfo(command_queue, param_name, param_value_size, param_value, param_value_size_ret)
	           : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL clFlush(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
	return r_clFlush ? r_clFlush(command_queue) : CL_INVALID_COMMAND_QUEUE;
}

CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void *pattern,
                                                    size_t pattern_size, size_t offset, size_t size,
                                                    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                                                    cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
	return r_clEnqueueFillBuffer ? r_clEnqueueFillBuffer(command_queue, buffer, pattern, pattern_size, offset, size,
	                                                     num_events_in_wait_list, event_wait_list, event)
	                             : CL_INVALID_COMMAND_QUEUE;
}

#endif // UFSECP_OPENCL_ENABLED
