#pragma once

#include "core.h"


/* CUDA kernel status checker for cuda-runtime API */
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = expr; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


    /* CUDA kernel status checker for cuda-driver API */
#define CUDA_DRV_CHECK(call) \
    do { \
        CUresult _r = (call); \
        if (_r != CUDA_SUCCESS) { \
            const char* _name = nullptr; \
            const char* _msg = nullptr; \
            cuGetErrorName(_r, &_name); \
            cuGetErrorString(_r, &_msg); \
            fprintf( \
                stderr, "[CUDA-DRIVER] %s failed: %s (%s)\n", \
                #call, _name ? _name : "UNKNOWN", _msg ? _msg : "no message" \
            ); \
        } \
    } while(0)


#define SETUP_PTX_KERNEL(sym) \
    extern unsigned char _ptx_##sym##_start[]; \
    extern unsigned char _ptx_##sym##_end[]; \
    \
    inline void load_ptx_kernel_##sym( \
        CUmodule* mod, \
        CUfunction* func, \
        const char* kernel_name \
    ) { \
        size_t size = _ptx_##sym##_end - _ptx_##sym##_start; \
        DEBUG("Loading binary _ptx_" << #sym << ", with size: " << size); \
        /* Convert to C-str, to avoid misplaced null-termination */\
        std::string ptx(reinterpret_cast<const char*>(_ptx_##sym##_start), size); \
        ptx.push_back('\0'); \
        CUDA_DRV_CHECK( \
            cuModuleLoadDataEx(mod, ptx.c_str(), 0, NULL, NULL) \
        ); \
        CUDA_DRV_CHECK( \
            cuModuleGetFunction(func, *mod, kernel_name) \
        ); \
    }


#define INIT_CUDA_DRIVER_CONTEXT(sym) \
    CUdevice device; \
    CUcontext context; \
    CUmodule module; \
    CUfunction kernel; \
    CUDA_DRV_CHECK(cuInit(0)); \
    cuDeviceGet(&device, 0); \
    cuCtxCreate(&context, 0, device); \
    load_ptx_kernel_##sym(&module, &kernel, #sym);


#define CLEANSE_CUDA_DRIVER_CONTEXT() \
    cuModuleUnload(module); \
    cuCtxDestroy(context);


namespace cuop {

    TensorPtr add(TensorPtr& a, TensorPtr& b);
    TensorPtr mm(TensorPtr& a, TensorPtr& b);
    TensorPtr mm2(TensorPtr& a, TensorPtr& b, string kernel);

}