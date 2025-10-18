#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "function/function.h"
#include "utils/utils.h"

using FunctionPtr = std::shared_ptr<Function>;


struct ComputationalGraph {
    using DirectedEdge = std::tuple<
        std::string,  // name of parent fn
        std::string,  // name of child fn
        std::vector<size_t>  // shape of intermediate tensor
    >;

    Counter<std::string> counter = Counter<std::string>{};
    std::vector<DirectedEdge> edges = std::vector<DirectedEdge>{};

    void add_edges(const TensorPtrVec& inputs, const FunctionPtr node);
};
