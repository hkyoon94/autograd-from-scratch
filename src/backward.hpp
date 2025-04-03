#include <torch/extension.h>

// g: incoming gradient
// x: forward pass result

torch::Tensor sigmoid_backward(torch::Tensor& g, torch::Tensor& x);

torch::Tensor mmul_left_backward(torch::Tensor& g, torch::Tensor& x);

torch::Tensor mmul_right_backward(torch::Tensor& g, torch::Tensor& x);

torch::Tensor add_broadcast_backward(torch::Tensor& x);

torch::Tensor ce_softmax_mean_backward(
    torch::Tensor& y,
    torch::Tensor& h,
    int N
);
