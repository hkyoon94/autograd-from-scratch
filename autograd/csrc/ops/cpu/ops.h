#pragma once

#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>

#include "tensor/tensor.h"


namespace py = pybind11;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrVec = std::vector<TensorPtr>;


namespace op {

    /* Initializers */
    TensorPtr zeros(const py::list shape);
    TensorPtr _zeros(const std::vector<size_t>& shape);
    TensorPtr zeros_like(const TensorPtr& tensor);
    TensorPtr ones(const py::list shape);
    TensorPtr ones_like(const TensorPtr& tensor);
    TensorPtr randn(const py::list shape, const float std);

    /* Operators */
    // scalar-div
    TensorPtr div(const TensorPtr& x, const float divisor);
    TensorPtrVec div_backward(const TensorPtr& g, const float divisor);
    // sum
    TensorPtr sum(const TensorPtr& x, const std::vector<size_t>& dims);
    TensorPtrVec sum_backward(const TensorPtr& g, const TensorPtr& x);
    // add
    TensorPtr add(const TensorPtr& a, const TensorPtr& b);
    TensorPtrVec add_backward(const TensorPtr& g, const TensorPtr& a, const TensorPtr& b);
        // only for grad-accumulation
    void add_inplace_contiguous(TensorPtr& a, const TensorPtr& b, float alpha = 1.0f);  
    // matmul
    TensorPtr mm(const TensorPtr& a, const TensorPtr& b);
    TensorPtrVec mm_backward(const TensorPtr& g, const TensorPtr& a, const TensorPtr& b);
    // conv2d
    TensorPtr conv2d(
        const TensorPtr& x_, const TensorPtr& weight_, const TensorPtr& bias_,
        const std::vector<uint>& s, const std::vector<uint>& d, const std::vector<uint>& p
    );
    TensorPtrVec conv2d_backward(
        const TensorPtr& g, const TensorPtr& x_, const TensorPtr& weight_,
        const std::vector<uint>& ctx
    );

    /* Element-wise activations */
    //sigmoid
    TensorPtr sigmoid(const TensorPtr& x);
    TensorPtrVec sigmoid_backward(const TensorPtr& g, const TensorPtr& x);
    //relu
    TensorPtr leaky_relu(const TensorPtr& x);
    TensorPtrVec leaky_relu_backward(const TensorPtr& g, const TensorPtr& x);
    // Dropout
    TensorPtr drop(const TensorPtr& x, const TensorPtr& mask);
    TensorPtrVec drop_backward(const TensorPtr& g, const TensorPtr& mask);

    /* Non-linear */
    TensorPtrVec ce_softmax_mean(const TensorPtr& x, const TensorPtr& y);
    TensorPtrVec ce_softmax_mean_backward(
        const TensorPtr& g, const TensorPtr& sm, const TensorPtr& y
    );
};
