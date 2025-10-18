#include <engine/graph.h>
#include <tensor/tensor.h>


void ComputationalGraph::add_edges(const TensorPtrVec& inputs, const FunctionPtr node) {
    std::string node_name = node->name() + "_" + std::to_string(node->id());
    for (TensorPtr tensor_ptr : inputs) {
        if (!tensor_ptr->requires_grad_) {
            if (!tensor_ptr->grad_fn_) {  // i.e., Non-param leaf tensor
                this->edges.emplace_back(
                    std::make_tuple(
                        "Non-Param_" + std::to_string(tensor_ptr->id()),
                        node_name,
                        tensor_ptr->shape_
                    )
                );
            } else {  // i.e., intermediate tensor derived from another fn
                const FunctionPtr parent = tensor_ptr->grad_fn_;
                this->edges.emplace_back(
                    std::make_tuple(
                        parent->name() + "_" + std::to_string(parent->id()),
                        node_name,
                        tensor_ptr->shape_
                    )
                );
            }
        }
        else {  // i.e., Param leaf-tensor
            DEBUG_ASSERT(!tensor_ptr->grad_fn_);
            this->counter["Param"] += 1;
            this->edges.emplace_back(
                std::make_tuple(
                    "Param_" + std::to_string(tensor_ptr->id()),
                    node_name,
                    tensor_ptr->shape_
                )
            );
        }
    }
}