#include "functional.h"
#include "core.h"
#include "ops_c.h"
#include "ops_common.h"
#include "utils.h"
#include <memory>


using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrVec = std::vector<TensorPtr>;
using FunctionPtr = std::shared_ptr<Function>;


// Functions
// view
std::string View::name() const { return this->_name_; }

TensorPtr View::forward(const TensorPtrVec& inputs, const std::vector<size_t>& as_shape) {    
    DEBUG("Performing view_forward, inputs: " << inputs[0]);
    // TODO: add dispatcher
    TensorPtr out = op::view(inputs[0], as_shape);

    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<View>();
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0]};
        new_node->original_dim_ = inputs[0]->shape_;  // TODO: enhance context type
        out->grad_fn_ = new_node;  // setting grad_fn
        DEBUG("out.requires_grad: " << out->requires_grad_);
        DEBUG("out.grad_fn" << out->grad_fn_);
    }
    return out;
}

TensorPtrVec View::backward(const TensorPtr& grad) const {
    DEBUG("Performing view_backward, in_grad: " << grad);
    // TODO: add dispatcher
    return {op::view(grad, this->original_dim_)};
}


// div
std::string Div::name() const { return this->_name_; }

TensorPtr Div::forward(const TensorPtrVec& inputs, const std::vector<float>& divisor) {    
    DEBUG("Performing div_forward, inputs: " << inputs[0]);
    // TODO: add dispatcher
    TensorPtr out = op::div(inputs[0], divisor[0]);

    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<Div>();
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0]};
        new_node->ctx_ = {Tensor::create(  // caching inputs for backward
            std::vector<size_t>{1},
            divisor[0]
        )};
        out->grad_fn_ = new_node;  // setting grad_fn
        DEBUG("out.requires_grad: " << out->requires_grad_);
        DEBUG("out.grad_fn" << out->grad_fn_);
    }
    return out;
}

TensorPtrVec Div::backward(const TensorPtr& grad) const {
    DEBUG("Performing div_backward, in_grad: " << grad);
    // TODO: add dispatcher
    return op::div_backward(grad, this->ctx_[0]->data<const float>()[0]);
}


// sum
std::string Sum::name() const { return this->_name_; }

TensorPtr Sum::forward(const TensorPtrVec& inputs, const std::vector<size_t>& dims) {    
    DEBUG("Performing sum_forward, input: " << inputs[0]);
    // TODO: add dispatcher
    // *** TODO: Test broadcast backward throughly, seem to be a bug exists in certain shape cases.
    TensorPtr out = op::sum(inputs[0], dims);
    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<Sum>();
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0]};  // caching inputs for backward
        new_node->original_dim_ = {2};  // dim to expand  // TODO: only for test
        out->grad_fn_ = new_node;  // setting grad_fn
        DEBUG("out.requires_grad: " << out->requires_grad_);
        DEBUG("out.grad_fn" << out->grad_fn_);
    }
    return out;
}

TensorPtrVec Sum::backward(const TensorPtr& grad) const {
    DEBUG(
        "Performing sum_backward, in_grad: " << grad <<
        ", target shape: " << parents_[0]->shape_
    );
    // TODO: add dispatcher
    return op::sum_backward(grad, this->parents_[0]);
}


// add
std::string Add::name() const { return this->_name_; }

TensorPtr Add::forward(const TensorPtrVec& inputs) {
    DEBUG("Performing add_forward, inputs: " << inputs[0] << ", " << inputs[1]);
    // TODO: add dispatcher
    TensorPtr out = op::add(inputs[0], inputs[1]);

    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<Add>();
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0], inputs[1]};
        out->grad_fn_ = new_node;
        DEBUG("out.requires_grad: " << out->requires_grad_);
        DEBUG("out.grad_fn" << out->grad_fn_);
    }
    return out;
}

TensorPtrVec Add::backward(const TensorPtr& grad) const {
    DEBUG("Performing Add backward, in_grad: " << grad);
    return op::add_backward(grad, this->parents_[0], this->parents_[1]);
};


// mm
std::string MatMul::name() const { return this->_name_; }

TensorPtr MatMul::forward(const TensorPtrVec& inputs) {
    DEBUG("Performing MatMul forward");   
    // TODO: add dispatcher
    TensorPtr out = op::mm(inputs[0],  inputs[1]);

    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<MatMul>();
        DEBUG("new_node name: " << new_node->_name_);
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0], inputs[1]};
        out->grad_fn_ = new_node;
        DEBUG("out.requires_grad: " << out->requires_grad_);
        DEBUG("out.grad_fn" << out->grad_fn_);
    }
    return out;
}

TensorPtrVec MatMul::backward(const TensorPtr& grad) const {
    DEBUG("Performing MatMul backward");
    // TODO: add dispatcher
    // TODO: implement lambda lazy implementation
    return op::mm_backward(grad, this->parents_[0], this->parents_[1]);
};


// Sigmoid
std::string Sigmoid::name() const { return this->_name_; }

TensorPtr Sigmoid::forward(const TensorPtrVec& inputs) {
    DEBUG("Performing Sigmoid forward");
    TensorPtr out = op::sigmoid(inputs[0]);
    
    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<Sigmoid>();
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0]};
        out->grad_fn_ = new_node;
        DEBUG("out.requires_grad: " << out->requires_grad_);
        DEBUG("out.grad_fn" << out->grad_fn_);
    }
    return out;
}

TensorPtrVec Sigmoid::backward(const TensorPtr& grad) const {
    return op::sigmoid_backward(grad, this->parents_[0]);
}


// LeakyRelu
std::string LeakyRelu::name() const { return this->_name_; }

TensorPtr LeakyRelu::forward(const TensorPtrVec& inputs) {
    DEBUG("Performing LeakyRelu forward");
    TensorPtr out = op::leaky_relu(inputs[0]);
    
    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<LeakyRelu>();
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0]};
        out->grad_fn_ = new_node;
        DEBUG("out.requires_grad: " << out->requires_grad_);
        DEBUG("out.grad_fn" << out->grad_fn_);
    }
    return out;
}

TensorPtrVec LeakyRelu::backward(const TensorPtr& grad) const {
    return op::leaky_relu_backward(grad, this->parents_[0]);
}


// Softmax -> CrossEntropy -> Mean fused
std::string SoftmaxCrossEntropyMean::name() const { return this->_name_; }

TensorPtr SoftmaxCrossEntropyMean::forward(const TensorPtrVec& inputs) {
    DEBUG("Performing CELoss forward");
    // inputs: { x, label }
    // out: { loss, softmax(x) }
    TensorPtrVec out = op::ce_softmax_mean(inputs[0], inputs[1]);

    // autograd posthook
    if (AutogradEngine::on_) {
        auto new_node = Function::create<SoftmaxCrossEntropyMean>();
        if (AutogradEngine::track_graph_) {
            AutogradEngine::graph_.add_edges(inputs, new_node);
        }
        new_node->parents_ = {inputs[0]};  // navigating parent tensor
        new_node->ctx_ = {
            out[1],  // softmax result
            inputs[1],  // label
        };
        out[0]->grad_fn_ = new_node;  // setting grad_fn
        DEBUG("out.requires_grad: " << out[0]->requires_grad_);
        DEBUG("out.grad_fn" << out[0]->grad_fn_);
    }
    return out[0];
}

TensorPtrVec SoftmaxCrossEntropyMean::backward(const TensorPtr& grad) const {
    return {
        op::ce_softmax_mean_backward(grad, ctx_[0], ctx_[1])
    };
}
