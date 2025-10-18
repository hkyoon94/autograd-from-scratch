#include <stdexcept>

#include "tensor/tensor.h"


/************************** Metadata view ops *************************/

namespace op {

// view
TensorPtr view(const TensorPtr& x, const std::vector<size_t>& as_shape) {
    size_t numel_ = calc_numel(as_shape);
    if (numel_ != x->numel_)
        throw std::runtime_error("view: numel mismatch");
    //! Should not directly copy x, which is a pointer of a Tensor here.
    //! The autograd chain breaks via infinite self-reference.
    // Thus we copy the object x itself, and should return the shared_ptr of it.
    Tensor out_tensor = *x;
    out_tensor.shape_ = as_shape;
    out_tensor.stride_ = calc_contiguous_stride(as_shape);
    return std::make_shared<Tensor>(out_tensor);
}


// TODO: transpose -> just change stride. If necessary, call contiguous.

};
