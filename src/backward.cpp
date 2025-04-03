// #include <vector>

#include "backward.hpp"
#include "threadpool.hpp"


static int num_threads = 4;
static ThreadPool pool(num_threads, false);  // pre-launching threads

// namespace py = pybind11;


// X, Y: (B, I)
// -> Z: (B, I)
// element-wise multiplication
void _sigmoid_backward(
    const float* X, const float* Y, float* Z, int Bs, int Be, int I
) {
    int bi;
    float sig_y;
    for (int b = Bs; b < Be; ++b) {  // batch parallel
        for (int i = 0; i < I; ++i) {
            bi = b*I + i; 
            sig_y = (1 / (1 + exp(-Y[bi]))); 
            Z[bi] = X[bi] * sig_y * (1 - sig_y);
        }
    }
}

// g: (B, I), x: (B, I) -> (B, I)
torch::Tensor sigmoid_backward(torch::Tensor& g, torch::Tensor& x) {
    g = g.contiguous();
    x = x.contiguous();

    int B = g.size(0);
    int I = g.size(1);
    torch::Tensor z = torch::zeros({B, I}, torch::kFloat32);

    const float* G = g.data_ptr<float>();
    const float* X = x.data_ptr<float>();
    float* Z = z.data_ptr<float>();

    // std::vector<std::function<void()>> tasks(num_threads);  // for multithreading
    // // int batch_per_thread = B / num_threads;
    // int batch_per_thread = (B < num_threads) ? B: B / num_threads;

    // for (int i = 0; i < num_threads; ++i) {
    //     int Bs = i * batch_per_thread;
    //     int Be = (i == num_threads - 1) ? B : Bs + batch_per_thread;
    //     tasks[i] = [&G, &X, &Z, Bs, Be, I] {
    //         _sigmoid_backward(G, X, Z, Bs, Be, I);
    //     };
    // }
    // py::gil_scoped_release release;
    // pool.push(tasks);
    // py::gil_scoped_acquire acquire;
    _sigmoid_backward(G, X, Z, 0, B, I);
    return z;
}

// X: (B, J)
// Y: (I, J)
// -> Z: (B, I)
// trailing dim이 j이기 때문에, 그대로 b -> i -> j 루프 순서 사용
void _bj_ij_bi(const float* X, const float* Y, float* Z, int Bs, int Be, int J, int I) {
    float a;
    for (int b = Bs; b < Be; ++b) {  // batch parallel
        for (int i = 0; i < I; ++i) {
            a = 0.0;
            for (int j = 0; j < J; ++j) {
                a += X[b*J + j] * Y[i*J + j];
            }
            Z[b*I + i] = a;
        }
    }
}

// g: (B, J), x: (I, J) -> (B, I)
torch::Tensor mmul_left_backward(torch::Tensor& g, torch::Tensor& x) {
    g = g.contiguous();
    x = x.contiguous();

    int B = g.size(0);
    int J = g.size(1);
    int I = x.size(0);
    torch::Tensor z = torch::zeros({B, I}, torch::kFloat32);

    const float* G = g.data_ptr<float>();
    const float* X = x.data_ptr<float>();
    float* Z = z.data_ptr<float>();

    // std::vector<std::function<void()>> tasks(num_threads);  // for multithreading
    // // int batch_per_thread = B / num_threads;
    // int batch_per_thread = (B < num_threads) ? B: B / num_threads;

    // for (int i = 0; i < num_threads; ++i) {
    //     int Bs = i * batch_per_thread;
    //     int Be = (i == num_threads - 1) ? B : Bs + batch_per_thread;
    //     tasks[i] = [&G, &X, &Z, Bs, Be, J, I] {
    //         _bj_ij_bi(G, X, Z, Bs, Be, J, I);
    //     };
    // }
    // py::gil_scoped_release release;
    // pool.push(tasks);
    // py::gil_scoped_acquire acquire;
    _bj_ij_bi(G, X, Z, 0, B, J, I);
    return z;
}

// X: (B, J)
// Y: (B, I)
// Z: (I, J)
// trailing dim이 B가 되면 cache miss로 성능이 떨어지므로
// cache locality를 위해, b -> i -> j 루프 순서 사용
void _bj_bi_ij(const float* X, const float* Y, float* Z, int Bs, int Be, int J, int I) {
    float y;
    for (int b = Bs; b < Be; ++b) {  // batch parallel
        for (int i = 0; i < I; ++i) {
            y = Y[b*I + i]; // Y[b, i]
            for (int j = 0; j < J; ++j) {
                Z[i*J + j] += X[b*J + j] * y;  // Z[i, j] += X[b, j] * Y[b, i]
            }
        }
    }
}

// g: (B, J), x: (B, I) -> (I, J)
torch::Tensor mmul_right_backward(torch::Tensor& g, torch::Tensor& x) {
    g = g.contiguous();
    x = x.contiguous();

    int B = g.size(0);
    int J = g.size(1);
    int I = x.size(1);
    torch::Tensor z = torch::zeros({I, J}, torch::kFloat32);

    const float* G = g.data_ptr<float>();
    const float* X = x.data_ptr<float>();
    float* Z = z.data_ptr<float>();

    // std::vector<std::function<void()>> tasks(num_threads);  // for multithreading
    // // int batch_per_thread = B / num_threads;
    // int batch_per_thread = (B < num_threads) ? B: B / num_threads;

    // for (int i = 0; i < num_threads; ++i) {
    //     int Bs = i * batch_per_thread;
    //     int Be = (i == num_threads - 1) ? B : Bs + batch_per_thread;
    //     tasks[i] = [&G, &X, &Z, Bs, Be, J, I] {
    //         _bj_bi_ij(G, X, Z, Bs, Be, J, I);
    //     };
    // }
    // py::gil_scoped_release release;
    // pool.push(tasks);
    // py::gil_scoped_acquire acquire;
    _bj_bi_ij(G, X, Z, 0, B, J, I);
    return z;
}

// X: (B, I)
// -> Y: (I)
// batch summation
void _bi_i(const float* X, float* Y, int Bs, int Be, int I) {
    for (int b = Bs; b < Be; ++b) {  // batch parallel
        for (int i = 0; i < I; ++i) {
            Y[i] += X[b*I + i];
        }
    }
}

// x: (B, I) -> (I)
torch::Tensor add_broadcast_backward(torch::Tensor& x) {
    x = x.contiguous();

    int B = x.size(0);
    int I = x.size(1);
    torch::Tensor y = torch::zeros({I}, torch::kFloat32);

    const float* X = x.data_ptr<float>();
    float* Y = y.data_ptr<float>();

    // std::vector<std::function<void()>> tasks(num_threads);  // for multithreading
    // // int batch_per_thread = B / num_threads;
    // int batch_per_thread = (B < num_threads) ? B: B / num_threads;

    // for (int i = 0; i < num_threads; ++i) {
    //     int Bs = i * batch_per_thread;
    //     int Be = (i == num_threads - 1) ? B : Bs + batch_per_thread;
    //     tasks[i] = [&X, &Y, Bs, Be, B, I] {
    //         _bi_i(X, Y, Bs, Be, I);
    //     };
    // }
    // py::gil_scoped_release release;
    // pool.push(tasks);
    // py::gil_scoped_acquire acquire;
    _bi_i(X, Y, 0, B, I);
    return y;
}

// Y, H: (B, I)
// -> Z: (B, I)
void _ce_softmax_mean_backward(
    const float* Y, const float* H, float* Z, int Bs, int Be, int I, int N
) {
    int bi;
    for (int b = Bs; b < Be; b++) {  // batch parallel
        for (int i = 0; i < I; i++) {
            bi = b*I + i;
            Z[bi] = (H[bi] - Y[bi]) / float(N);
        }
    }
}

// y: one-hot label (B, I)
// h: logits (B, I)
// N: # batches
torch::Tensor ce_softmax_mean_backward(
    torch::Tensor& y,
    torch::Tensor& h,
    int N
) {
    y = y.contiguous();
    h = h.contiguous();

    int B = h.size(0);
    int I = h.size(1);
    torch::Tensor z = torch::zeros({B, I}, torch::kFloat32);

    const float* Y = y.data_ptr<float>();
    const float* H = h.data_ptr<float>();
    float* Z = z.data_ptr<float>();

    // std::vector<std::function<void()>> tasks(num_threads);  // for multithreading
    // // int batch_per_thread = B / num_threads;
    // int batch_per_thread = (B < num_threads) ? B: B / num_threads;

    // for (int i = 0; i < num_threads; ++i) {
    //     int Bs = i * batch_per_thread;
    //     int Be = (i == num_threads - 1) ? B : Bs + batch_per_thread;
    //     tasks[i] = [&Y, &H, &Z, Bs, Be, I, N] {
    //         _ce_softmax_mean_backward(Y, H, Z, Bs, Be, I, N);
    //     };
    // }
    // py::gil_scoped_release release;
    // pool.push(tasks);
    // py::gil_scoped_acquire acquire;
    _ce_softmax_mean_backward(Y, H, Z, 0, B, I, N);
    return z;
}
