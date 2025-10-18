#pragma once

#include "tensor/tensor.h"


namespace cuop {

    TensorPtr add(TensorPtr& a, TensorPtr& b);
    TensorPtr mm(TensorPtr& a, TensorPtr& b);
    TensorPtr mm2(TensorPtr& a, TensorPtr& b, std::string kernel);

}