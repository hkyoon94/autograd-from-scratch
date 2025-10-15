constexpr int SIZE_T = 32;


#ifdef __CUDACC__

__global__ void mmul_naive(
    const float* ptr_a,
    const float* ptr_b,
    float* ptr_c,
    const int M, const int N, const int K
) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M || j >= N) return;

    float acc = 0.0;
    for (int k = 0; k < K; k++) {
        acc += ptr_a[i * K + k] * ptr_b[k * N + j];
    }
    ptr_c[i * N + j] = acc;
}


__global__ void mmul_tiled(
    const float* __restrict__ ptr_a,
    const float* __restrict__ ptr_b,
    float* __restrict__ ptr_c,
    const int M, const int N, const int K
) {
    int i_local = threadIdx.y;
    int j_local = threadIdx.x;
    int i_off = blockDim.y * blockIdx.y;
    int j_off = blockDim.x * blockIdx.x;
    int i = i_off + i_local;
    int j = j_off + j_local;

    __shared__ float sa[SIZE_T][SIZE_T + 1];  // 4*16*16=1024 (1KB)
    __shared__ float sb[SIZE_T][SIZE_T + 1];  // 4*16*16=1024 (1KB)

    const int num_k_tiles = (K + SIZE_T - 1) / SIZE_T;

    float acc = 0.0;
    for (int k_ = 0; k_ < num_k_tiles; k_++) {
        // loading A tile & B tile to shared memory
        int k_off = k_ * SIZE_T;
        int a_col = k_off + j_local;  // for A[i, k_off + j_local]
        int b_row = k_off + i_local;  // for B[k_off + i_local, j]
        float a, b;
        if (a_col >= K || i >= M)  // Removing tile edge
            a = 0.0;
        else
            a = ptr_a[i * K + a_col];
        if (b_row >= K || j >= N)  // Removing tile edge
            b = 0.0;
        else
            b = ptr_b[b_row * N + j];

        sa[i_local][j_local] = a;
        sb[i_local][j_local] = b;
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < SIZE_T; k++) {
            acc += sa[i_local][k] * sb[k][j_local];
            // no bank conflict, since thread index increases in j_local
        }
        __syncthreads();
    }
    if (i >= M || j >= N) {  // If not tile edge
        return;
    }
    ptr_c[i * N + j] = acc;
}

#endif

#include <cublas_v2.h>

#include "core.h"
#include "ops_c.h"
#include "ops_cuda.h"
#include "utils.h"

// SETUP_PTX_KERNEL(add);
// SETUP_PTX_KERNEL(mm);  // TODO: Check triton kernel integration


namespace cuop {

TensorPtr add(TensorPtr& a, TensorPtr& b) {
    // INIT_CUDA_DRIVER_CONTEXT(add);  // TODO: Enable compile caching
    DEBUG("Performing CUDA add with input0: " << a->shape_ << ", input1: " << b->shape_);
    size_t N = a->numel_;
    assert(N == b->numel_);
    assert(a->device_ == Device::CUDA && b->device_ == Device::CUDA);

    TensorPtr c = Tensor::create(a->shape_, a->dtype_, Device::CUDA);
    const float* a_ptr = a->data<const float>();
    const float* b_ptr = b->data<const float>();
    float* c_ptr = c->data<float>();
    void* dummy_null = 0;

    int blockDimX = 1024, blockDimY = 1, blockDimZ = 1;
    int gridDimX = (N + blockDimX - 1) / blockDimX, gridDimY = 1, gridDimZ = 1;

    void* args[] = { &a_ptr, &b_ptr, &c_ptr, &N, &dummy_null };
    // CUDA_DRV_CHECK(
    //     cuLaunchKernel(
    //         kernel,
    //         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
    //         0, 0, args, 0
    //     )
    // );
    cuCtxSynchronize();
    DEBUG("cuop::add end");
    return c;
}


TensorPtr mm(TensorPtr& a, TensorPtr& b) {
    // INIT_CUDA_DRIVER_CONTEXT(mm);
    DEBUG("Performing CUDA matmul with input0: " << a->shape_ << ", input1: " << b->shape_);

    size_t M = a->shape_[0];
    size_t K = a->shape_[1];
    size_t N = b->shape_[1];

    assert(K == b->shape_[0]);
    assert(a->device_ == Device::CUDA && b->device_ == Device::CUDA);

    TensorPtr c = Tensor::create(std::vector{M, N}, a->dtype_, Device::CUDA);
    const float* a_ptr = a->data<const float>();
    const float* b_ptr = b->data<const float>();
    float* c_ptr = c->data<float>();

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* here, we use the following facts:
        1. transpose of row-major equals to colum-major layout.
        2. For A @ B = C, we have B^T @ A^T = C^T,
            which means inverted multiplication of colum-major layout of row-majors
            yields transposed colum-major layout, i.e., thus same as row-major.  
    */
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        b_ptr, N,
        a_ptr, K,
        &beta,
        c_ptr, N
    );
    cublasDestroy(handle);
    return c;

    // TODO: Check triton kernel integration
    /*
    int blockDimX = 256, blockDimY = 1, blockDimZ = 1;
    int gridDimX = 128;
    int gridDimY = 1;
    int gridDimZ = 1;

    DEBUG(a->stride_[0] << " " << b->stride_[0] << " " << c->stride_[0]);

    void* args[] = {
        &a_ptr, &b_ptr, &c_ptr,
        &M, &N, &K,
        &a->stride_[0],
        &b->stride_[0],
        &c->stride_[0],
        &dummy_null
    };
    CUDA_DRV_CHECK(
        cuLaunchKernel(
            kernel,
            gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
            0, 0, args, 0
        )
    );
    float* h_c = (float*)malloc(c->numel_ * sizeof(float));
    cudaDeviceSynchronize();
    DEBUG("cuop::mm end");
    */
    return c;
}


// TODO: only for testing
TensorPtr mm2(TensorPtr& a, TensorPtr& b, string kernel) {
    size_t M = a->shape_[0];
    size_t N = b->shape_[1];
    size_t K = a->shape_[1];

    TensorPtr c = Tensor::create(vector{M, N}, 0.0f);

    const float* ptr_a = a->data<const float>();
    const float* ptr_b = b->data<const float>();
    float* ptr_c = c->data<float>();

    if (kernel == "naive") {
        dim3 gridDim((int)((N + SIZE_T - 1) / SIZE_T), (int)((M + SIZE_T - 1) / SIZE_T));
        dim3 blockDim(SIZE_T, SIZE_T);
        mmul_naive<<<gridDim, blockDim>>>(ptr_a, ptr_b, ptr_c, M, N, K);
    }
    else if (kernel == "shared") {
        dim3 gridDim((int)((N + SIZE_T - 1) / SIZE_T), (int)((M + SIZE_T - 1) / SIZE_T));
        dim3 blockDim(SIZE_T, SIZE_T);
        mmul_tiled<<<gridDim, blockDim>>>(ptr_a, ptr_b, ptr_c, M, N, K);
    }
    cudaDeviceSynchronize();
    return c;
}


}  // namespace cuop;
