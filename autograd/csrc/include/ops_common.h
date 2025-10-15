#pragma once

#include <vector>

#include "core.h"


inline std::vector<size_t> _broadcast_shapes(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b
) {
    size_t na = a.size();
    size_t nb = b.size();
    size_t n  = std::max(na, nb);

    std::vector<size_t> out(n);

    // 뒤에서부터 비교
    for (size_t i = 0; i < n; i++) {
        size_t dim_a = (i < na) ? a[na - 1 - i] : 1;
        size_t dim_b = (i < nb) ? b[nb - 1 - i] : 1;

        if (dim_a == dim_b || dim_a == 1 || dim_b == 1)
            out[n - 1 - i] = std::max(dim_a, dim_b);
        else
            throw std::runtime_error("broadcast_shapes: incompatible shapes");
    }
    DEBUG("Got shape1 & shape2: " << a << ", " << b << ", Result: " << out);
    return out;
}


inline std::vector<size_t> _dims_to_reduce(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& target_shape
) {
    // 어떤 축을 줄였는지 계산
    std::vector<size_t> dims;
    int ndim_diff = shape.size() - target_shape.size();

    for (int i = 0; i < ndim_diff; i++)
        dims.push_back(i);
    
    for (int i = 0; i < (int)target_shape.size(); i++) {
        if (shape[ndim_diff + i] != target_shape[i]) {
            if (target_shape[i] == 1)
                dims.push_back(ndim_diff + i);
            else
                throw std::runtime_error("reduce_sum_to_shape: incompatible");
        }
    }
    DEBUG("Got shape: " << shape << ", Result: " << dims);
    return dims;
}


inline void squeeze_shape(std::vector<size_t>& shape) {
    shape.erase(
        std::remove_if(
            shape.begin(),
            shape.end(),
            [](size_t x) { return x == 1; }
        ),
        shape.end()
    );
}


namespace op {

    /* View-manipulations */
    // view
    TensorPtr view(const TensorPtr& x, const std::vector<size_t>& as_shape);

    // index-select
    Tensor index_select(TensorPtr& x, std::vector<size_t>& idxs);
    TensorPtrVec index_select_backward(TensorPtr& grad);

}
