#include <cstdlib>
#include <cstddef>
#include <random>
#include <vector>

#include "tensor/tensor.h"
#include "ops/ops_common.h"
#include "utils/utils.h"


// This machine is i7-14700kf, thus only supports _mm256_
constexpr size_t REGISTER_SIZE = 8;

using TensorPtr = std::shared_ptr<Tensor>;

// Kernel APIs
namespace op {

/******************************* Tensor initializers *********************************/

TensorPtr _zeros(const std::vector<size_t>& shape) {
    return Tensor::create(shape, 0.0f);
}
TensorPtr zeros(const py::list shape) {
    std::vector<size_t> shape_;
    for (auto item : shape) {
        shape_.push_back(item.cast<size_t>());
    }
    return Tensor::create(shape_, 0.0f);
}

TensorPtr zeros_like(const TensorPtr& tensor) {
    return Tensor::create(tensor->shape_, 0.0f);
}

TensorPtr ones(const py::list shape) {
    std::vector<size_t> shape_;
    for (auto item : shape) {
        shape_.push_back(item.cast<size_t>());
    }
    return Tensor::create(shape_, 1.0f);
}

TensorPtr ones_like(const TensorPtr& tensor) {
    return Tensor::create(tensor->shape_, 1.0f);
}

TensorPtr randn(const py::list shape, const float std) {  // only for testing for now
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);

    std::vector<size_t> shape_;
    for (auto item : shape) {
        shape_.push_back(item.cast<size_t>());
    }
    TensorPtr z = Tensor::create(shape_);
    float* z_ptr = z->data<float>();
    for (size_t i = 0; i < z->numel_; i++) {
        z_ptr[i] = dist(gen);
    }
    return z;
}

/****************************** Operators *********************************/

void inline _masked_fill(float* Z, const float* X, const bool* M, const size_t N) {
    for (size_t i = 0; i < N; i++)
        Z[i] = (M[i] ? X[i] : 0.0f);
}
TensorPtr drop(const TensorPtr& x, const TensorPtr& mask) {
    TensorPtr x_view = x->contiguous();
    TensorPtr z = Tensor::create(x->shape_);
    _masked_fill(
        z->data<float>(),
        x_view->data<const float>(),
        mask->data<const bool>(),
        x_view->numel_
    );
    return z;
}
TensorPtrVec drop_backward(const TensorPtr& g, const TensorPtr& mask) {
    DEBUG_ASSERT(g->contiguous_);
    TensorPtr z = Tensor::create(g->shape_);
    _masked_fill(
        z->data<float>(),
        g->data<const float>(),
        mask->data<const bool>(),
        g->numel_
    );
    return {z};
}

// elementwise div
inline void _div(const float* X, float* Z, float divisor, size_t N) noexcept {
    for (size_t i = 0; i < N; i++) {
        Z[i] = X[i] / divisor;
    }
}
TensorPtr div(const TensorPtr& x, const float divisor) {
    assert(divisor != 0);
    TensorPtr x_view = x->contiguous();
    TensorPtr out = Tensor::create(x_view->shape_);
    float* out_ptr = out->data<float>();
    const float* x_ptr = x_view->data<const float>();
    _div(x_ptr, out_ptr, divisor, x->numel_);
    return out;
}

inline void _div_backward_inplace(float* G, float divisor, size_t N) noexcept {
    for (size_t i = 0; i < N; i++)
        G[i] /= divisor;
}
TensorPtrVec div_backward(const TensorPtr& g, const float divisor) {
    DEBUG_ASSERT(g->contiguous_);
    auto g_c = g->contiguous();
    // can safely inplace while backward
    float* g_ptr = g_c->data<float>();
    _div_backward_inplace(g_ptr, divisor, g->numel_);
    DEBUG("Div backward result: " << g);
    return {g_c};
}


// sum kernel
void _sum(
    const TensorPtr& x, TensorPtr& out, const std::vector<char>& reduce_flag
) {
    size_t ndim = x->shape_.size();
    float* out_ptr = out->data<float>();
    const float* x_ptr = x->data<const float>();

    size_t total = x->numel_;
    for (size_t idx = 0; idx < total; idx++) {
        // idx → 다차원 좌표
        size_t tmp = idx;
        size_t in_off = 0, out_off = 0;
        for (int d = ndim - 1; d >= 0; d--) {
            int coord = tmp % x->shape_[d];
            tmp /= x->shape_[d];

            in_off += coord * x->stride_[d];
            int ocoord = reduce_flag[d] ? 0 : coord;
            out_off += ocoord * out->stride_[d];
        }
        out_ptr[out_off] += x_ptr[in_off];
    }
}

// Element-summation
TensorPtr sum(const TensorPtr& x, const std::vector<size_t>& dims) {
    size_t ndim = x->shape_.size();
    std::vector<size_t> out_shape = x->shape_;
    std::vector<char> reduce_flag(ndim, 0);

    for (size_t ax : dims) {
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim) 
            throw std::runtime_error("sum_axes: invalid axis");
        reduce_flag[ax] = 1;
        out_shape[ax] = 1;
    }
    TensorPtr out = Tensor::create(out_shape, 0.0f);
    _sum(x, out, reduce_flag);
    // squeeze_shape(out->shape_);
    return out;
}

TensorPtrVec sum_backward(const TensorPtr& g, const TensorPtr& x) {
    DEBUG_ASSERT(g->contiguous_);
    TensorPtr g_b = g->broadcast_to(x->shape_)->contiguous();
    DEBUG("sum backward result: " << g_b << ", stride: " << g_b->stride_);
    return {g_b};
}


// add
void _add_contiguous(
    const float* X, const float* Y, float* Z, const size_t N
) noexcept {
    for (size_t i = 0; i < N; i++)
        Z[i] = X[i] + Y[i];
}
TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
    DEBUG("Performing broadcast enabled add");
    // output shape = broadcasted shape
    std::vector<size_t> out_shape = _broadcast_shapes(a->shape_, b->shape_);
    TensorPtr a_view = (a->shape_ != out_shape)
        ? a->broadcast_to(out_shape)->contiguous() : a;
    TensorPtr b_view = (b->shape_ != out_shape)
        ? b->broadcast_to(out_shape)->contiguous() : b;
    TensorPtr out = Tensor::create(out_shape);
    _add_contiguous(
        a_view->data<const float>(), 
        b_view->data<const float>(), 
        out->data<float>(), 
        out->numel_
    );
    return out;
}

TensorPtrVec add_backward(
    const TensorPtr& g, const TensorPtr& a, const TensorPtr& b
) {
    DEBUG_ASSERT(g->contiguous_);
    TensorPtrVec outs;
    outs.emplace_back(
        op::sum(g, _dims_to_reduce(g->shape_, a->shape_))
    );
    outs.emplace_back(
        op::sum(g, _dims_to_reduce(g->shape_, b->shape_))
    );
    DEBUG("Add backward result: " << outs[0] << ", " << outs[1]);
    return outs;
}

// only for gradient accumulation & optimizer step
void _add_inplace_contiguous(
    float* X, const float* Y, size_t L
) noexcept {
    for (size_t i = 0; i < L; i++) {
        X[i] += Y[i];
    }
}
void _add_inplace_contiguous_alpha(
    float* X, const float* Y, size_t L, float alpha
) noexcept {
    for (size_t i = 0; i < L; i++) {
        X[i] += alpha * Y[i];
    }
}
constexpr float ep = 1e-6;
void add_inplace_contiguous(TensorPtr& a, const TensorPtr& b, float alpha) {
    DEBUG(
        "Perfoming add_inplace operation, " << "a: " << a << 
        "b: " << b << " (inplace to a)"
    );
    size_t L = a->numel_;
    DEBUG_ASSERT(L == b->numel_);
    DEBUG_ASSERT(a->contiguous_ && b->contiguous_);
    if (abs(alpha - 1.0f) < ep)
        _add_inplace_contiguous(a->data<float>(), b->data<const float>(), L);
    else {
        _add_inplace_contiguous_alpha(
            a->data<float>(), b->data<const float>(), L, alpha
        );
    }
}


// X: (I, K)
// Y: (K, J)
// -> Z: (I, J)
// Fast implementation of A @ B, with virtual transpose
// only can be used when A, B is both contiguous
void _mmul_contiguous(
    const float* A,
    const float* B,
    float* C,
    size_t I,
    size_t K,
    size_t J
) noexcept {
    for (size_t i = 0; i < I; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float A_val = A[i * K + k];
            for (size_t j = 0; j < J; ++j) {
                C[i*J + j] += A_val * B[k*J + j];  // broadcasting A (Z needs zero init)
            }
        }
    }
}
void _mmul_stride(
    const float* A, const size_t* strideA,
    const float* B, const size_t* strideB,
    float* C,       const size_t* strideC,
    size_t I, size_t K, size_t J
) noexcept {
    for (size_t i = 0; i < I; ++i) {
        for (size_t k = 0; k < K; ++k) {
            // A_val = A[i, k]
            size_t offA = i*strideA[0] + k*strideA[1];
            float A_val = A[offA];
            for (size_t j = 0; j < J; ++j) {
                size_t offB = k*strideB[0] + j*strideB[1];
                size_t offC = i*strideC[0] + j*strideC[1];
                C[offC] += A_val * B[offB];
            }
        }
    }
}
TensorPtr mm(const TensorPtr& a, const TensorPtr& b) {
    DEBUG("Performing mm with input_0: " << a->shape_ << ", input_1: " << b->shape_);
    size_t I = a->shape_[0];
    size_t K = a->shape_[1];
    size_t J = b->shape_[1];
    DEBUG_ASSERT(K == b->shape_[0]);

    std::vector<size_t> out_shape = {I, J};
    TensorPtr out = Tensor::create(out_shape, 0.0f);  // zero init
    if (a->contiguous_ && b->contiguous_) {
        DEBUG("Using contiguous routine");
        _mmul_contiguous(
            a->data<const float>(), 
            b->data<const float>(),
            out->data<float>(),
            I, K, J
        );
    } else {
        DEBUG("Using strided routine");
        _mmul_stride(
            a->data<const float>(), a->stride_.data(),
            b->data<const float>(), b->stride_.data(),
            out->data<float>(), out->stride_.data(),
            I, K, J
        );
    }
    DEBUG("Result: " << out->__repr__());
    return out;
}


// X: (B, J)
// Y: (I, J)
// -> Z: (B, I)
// Fast implementation of X @ Y^t.
void _gradA_contiguous(
    const float* X, const float* Y, float* Z,
    size_t Bs, size_t Be, size_t J, size_t I
) noexcept {
    float a;
    for (size_t b = Bs; b < Be; ++b) {  // batch parallel
        for (size_t i = 0; i < I; ++i) {
            a = 0.0;
            for (size_t j = 0; j < J; ++j) {
                a += X[b*J + j] * Y[i*J + j];
            }
            Z[b*I + i] = a; // only malloc is enough
        }
    }
}
// g: (B, J), x: (I, J) -> grad @ x.T: (B, I)
// X: (B, J)
// Y: (B, I)
// Z: (I, J)
// Fast implementation of Y^t @ X.
// In order to prevent cache miss, we use b -> i -> j loop order.
void _gradB_contiguous(
    const float* X, const float* Y, float* Z,
    size_t Bs, size_t Be, size_t J, size_t I
) noexcept {
    float y;
    for (size_t b = Bs; b < Be; ++b) {  // batch parallel
        for (size_t i = 0; i < I; ++i) {
            y = Y[b*I + i]; // Y[b, i]
            for (size_t j = 0; j < J; ++j) {
                Z[i*J + j] += X[b*J + j] * y;  // Z[i, j] += X[b, j] * Y[b, i]
            }
        }
    }
}
// grad w.r.t A: g(I, J) @ B(K, J)^T = (I, K)
void _gradA_stride (
    const float* G, const size_t* stG,
    const float* B, const size_t* stB,
    float* dA, const size_t* stA,
    size_t I, size_t J, size_t K
) noexcept {
    for (size_t i = 0; i < I; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float acc = 0.f;
            for (size_t j = 0; j < J; ++j) {
                size_t offG = i * stG[0] + j * stG[1];
                size_t offB = k * stB[0] + j * stB[1]; // B[k,j]
                acc += G[offG] * B[offB];
            }
            size_t offA = i * stA[0] + k * stA[1];
            dA[offA] = acc;
        }
    }
}
// grad w.r.t B: A(I, K)^T @ g(I, J) = (K, J)
void _gradB_stride(
    const float* G, const size_t* stG,
    const float* A, const size_t* stA,
    float* dB, const size_t* stB,
    size_t I, size_t J, size_t K
) noexcept {
    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < J; ++j) {
            float acc = 0.f;
            for (size_t i = 0; i < I; ++i) {
                size_t offA = i * stA[0] + k * stA[1]; // A[i,k]
                size_t offG = i * stG[0] + j * stG[1]; // g[i,j]
                acc += A[offA] * G[offG];
            }
            size_t offB = k * stB[0] + j * stB[1];
            dB[offB] = acc;
        }
    }
}

TensorPtrVec mm_backward(const TensorPtr& g, const TensorPtr& a, const TensorPtr& b) {
    DEBUG_ASSERT(g->contiguous_);
    DEBUG(
        "Performing mm_backward_0 (g @ b.T) with input_0: " <<
        g->shape_ << ", input_1: " << b->shape_
    );
    size_t B = g->shape_[0];
    size_t J = g->shape_[1];
    size_t I = b->shape_[0];
    DEBUG_ASSERT(J == b->shape_[1]);

    std::vector<size_t> out_shape = {B, I};
    TensorPtr z1 = Tensor::create(out_shape, 0.0f);  // only malloc

    if (b->contiguous_) {
        DEBUG("Using contiguous routine");
        _gradA_contiguous(
            g->data<const float>(),
            b->data<const float>(),
            z1->data<float>(),
            0, B, J, I
        );
    } else {
        DEBUG("Using strided routine");
        _gradA_stride(
            g->data<const float>(), g->stride_.data(),
            b->data<const float>(), b->stride_.data(),
            z1->data<float>(), z1->stride_.data(),
            B, J, I
        );
    }
    DEBUG("Result: " << z1);
    DEBUG(
        "Performing mm_backward_1 (x.T @ g) with input_0: " <<
        g->shape_ << ", input_1: " << a->shape_
    );
    B = g->shape_[0];
    J = g->shape_[1];
    I = a->shape_[1];
    DEBUG_ASSERT(B == a->shape_[0]);

    out_shape = {I, J};
    TensorPtr z2 = Tensor::create(out_shape, 0.0f);
    if (a->contiguous_) {
        DEBUG("Using contiguous routine");
        _gradB_contiguous(
            g->data<const float>(),
            a->data<const float>(),
            z2->data<float>(),
            0, B, J, I
        );
    } else {
        DEBUG("Using strided routine");
        _gradB_stride(
            g->data<const float>(), g->stride_.data(),
            a->data<const float>(), a->stride_.data(),
            z2->data<float>(), z2->stride_.data(),
            B, J, I
        );
    }
    DEBUG("Result: " << z2);
    return {z1, z2};
}


/*******************************  Activations *********************************/

void _sigmoid_forward_contiguous(
    const float* X, float* Z, size_t i_start, size_t i_end
) noexcept {
    for (size_t i = i_start; i < i_end; i++)
        Z[i] = (1 / (1 + exp(-X[i])));
}
TensorPtr sigmoid(const TensorPtr& x) {
    DEBUG("Performing sigmoid forward with input_0: " << x->shape_);
    TensorPtr x_view = x->contiguous();
    size_t N = x->numel_;
    TensorPtr z = Tensor::create(x->shape_);
    _sigmoid_forward_contiguous(
        x_view->data<const float>(),
        z->data<float>(),
        0, N
    );
    return z;
}

// element-wise
void _sigmoid_backward(
    const float* G, const float* X, float* Z, size_t i_start, size_t i_end
) noexcept {
    float sig_x;
    for (size_t i = i_start; i < i_end; i++) {
        sig_x = (1 / (1 + exp(-X[i]))); 
        Z[i] = G[i] * sig_x * (1 - sig_x);  // for output Z, only malloc is enough
    }
}
TensorPtrVec sigmoid_backward(const TensorPtr& g, const TensorPtr& x) {
    DEBUG_ASSERT(g->contiguous_);
    DEBUG("Performing sigmoid_backward_0 with input_0: " << g->shape_);
    size_t N = g->numel_;
    assert(N == x->numel_);
    TensorPtr z = Tensor::create(x->shape_);  // only malloc
    _sigmoid_backward(
        g->data<const float>(),
        x->data<const float>(),
        z->data<float>(),
        0, N
    );
    return {z};
}


// relu
void _leaky_relu_forward_contiguous(
    const float* X, float* Z, size_t i_start, size_t i_end, float c
) noexcept {
    for (size_t i = i_start; i < i_end; i++) {
        Z[i] = (X[i] > 0.0f ? X[i] : c * X[i]);
    }
}
TensorPtr leaky_relu(const TensorPtr& x) {
    TensorPtr x_view = x->contiguous();
    DEBUG("Performing relu forward with input_0: " << x->shape_);
    size_t N = x->numel_;
    DEBUG("N: " << N);
    TensorPtr z = Tensor::create(x->shape_);  // only malloc
    DEBUG_ASSERT(x->numel_ == z->numel_);
    _leaky_relu_forward_contiguous(
        x_view->data<const float>(),
        z->data<float>(),
        0, N, 0.01f
    );
    return z;
}

void _leaky_relu_backward(
    const float* G, const float* X, float* Z, size_t i_start, size_t i_end, float c
) noexcept {
    for (size_t i = i_start; i < i_end; i++) {
        Z[i] = G[i] * (X[i] > 0.0f ? 1.0f : c);
    }
}
TensorPtrVec leaky_relu_backward(const TensorPtr& g, const TensorPtr& x) {
    DEBUG_ASSERT(g->contiguous_);
    DEBUG("Performing relu_backward_0 with input_0: " << g->shape_);
    size_t N = g->numel_;
    DEBUG_ASSERT(N == x->numel_);
    TensorPtr z = Tensor::create(g->shape_);  // only malloc
    _leaky_relu_backward(
        g->data<const float>(), 
        x->data<const float>(), 
        z->data<float>(), 
        0, N, 0.01f
    );
    return {z};
}


/*  Softmax-CrossEntropy-mean fused.
    params:
        X: input hidden states
        Y: (probably soft) one-hot encoded labels
        sm: float* for caching softmax(H)
        B, I = Y.shape
    X, Y: (B, I) -> Z: (B, I)
*/
float _ce_softmax_mean(
    const float* X, const float* Y, float* sm, size_t B, size_t I
) noexcept {
    double total = 0.0;
    double* row_exp_cache = (double*)malloc(I * sizeof(double));
    double _exp;
    for (size_t b = 0; b < B; ++b) {  // for each batch
        const float* y_b = Y + b * I;  // ptr Y[b, :]
        const float* x_b = X + b * I;  // ptr X[b, :]
        float* sm_b = sm + b * I;  // ptr sm[b, :]
        // compute row-wise max
        double m = x_b[0];
        for (size_t i = 1; i < I; ++i) {
            if (x_b[i] > m) m = x_b[i];
        }
        // compute safe sumexp (softmax denominator)
        double sum_exp = 0.0;
        for (size_t i = 0; i < I; ++i) {
            _exp = std::exp(double(x_b[i]) - m);
            row_exp_cache[i] = _exp;
            sum_exp += _exp;
        }
        // compute softmax
        for (size_t i = 0; i < I; ++i) {
            sm_b[i] = row_exp_cache[i] / sum_exp;
        }
        // 3) cross-entropy(y, softmax(x))
        double loss_b = 0.0;
        for (size_t i = 0; i < I; ++i) {
            loss_b += -double(y_b[i]) * std::log(sm_b[i]);
        }
        total += loss_b;
    }
    free(row_exp_cache);
    return float(total / double(B));
}
TensorPtrVec ce_softmax_mean(const TensorPtr& x, const TensorPtr& y) {
    DEBUG(
        "Performing ce_softmax_mean with shapes y: "
        << y->shape_ << ", h: " << x->shape_
    );
    assert(y->shape_ == x->shape_);
    assert(y->contiguous_ && x->contiguous_);
    size_t B = y->shape_[0];
    size_t I = y->shape_[1];
    // for caching result of softmax for backward
    TensorPtr sm = Tensor::create(x->shape_);  // only-malloc
    float res = _ce_softmax_mean(
        x->data<const float>(),
        y->data<const float>(),
        sm->data<float>(),
        B, I
    );
    return {Tensor::create(std::vector<size_t>{1}, res), sm};
}


// Softmax-CrossEntropy-mean fused backward
// Y, H: (B, I)
// -> Z: (B, I)
void _ce_softmax_mean_backward(
    const float g, const float* sm, const float* Y, float* Z, size_t B, size_t I
) noexcept {
    size_t bi;
    for (size_t b = 0; b < B; b++) {  // batch parallel
        for (size_t i = 0; i < I; i++) {
            bi = b * I + i;
            Z[bi] = g * (sm[bi] - Y[bi]) / B;
        }
    }
}
TensorPtrVec ce_softmax_mean_backward(
    const TensorPtr& g, const TensorPtr& sm, const TensorPtr& y
) {
    DEBUG_ASSERT(g->numel_ == 1);
    size_t B = sm->shape_[0];
    size_t I = sm->shape_[1];
    std::vector<size_t> out_shape = {B, I};
    TensorPtr z = Tensor::create(out_shape);
    _ce_softmax_mean_backward(
        g->data<float>()[0],
        sm->data<const float>(),
        y->data<const float>(),
        z->data<float>(), 
        B, I
    );
    DEBUG("Result: " << z);
    return {z};
}

};  // namespace 'op'
