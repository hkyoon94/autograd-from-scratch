#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "tensor/tensor.h"


using TensorPtr = std::shared_ptr<Tensor>;
using str = std::string;

struct IValue {
    using List = std::vector<IValue>;
    using Dict = std::unordered_map<std::string, IValue>;

    std::variant<
        std::monostate,
        int64_t,
        double,
        bool,
        str,
        TensorPtr,
        List,
        Dict
    > value;
};
