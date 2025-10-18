#include "engine/backward.h"
#include "ops/cpu/ops.h"

struct ComputationalGraph;


ComputationalGraph AutogradEngine::graph_{};
bool AutogradEngine::on_ = true;
bool AutogradEngine::track_graph_ = false;

void AutogradEngine::on(bool flag) {
    AutogradEngine::on_ = flag;
    #ifndef NDEBUG
        if (flag)
            DEBUG("Autograd ON");
        else
            DEBUG("Autograd OFF");
    #endif
}

void AutogradEngine::track_graph(bool flag) {
    AutogradEngine::track_graph_ = flag;
    AutogradEngine::graph_.edges.clear();
    #ifndef NDEBUG
        if (flag)
            DEBUG("Autograd now saves computational graph");
        else
            DEBUG("Autograd now don't saves computational graph");
    #endif
}

ComputationalGraph AutogradEngine::get_graph() {
    return AutogradEngine::graph_;
}

void AutogradEngine::clear_graph() {
    AutogradEngine::graph_.edges.clear();
}

/* Below implementation is exactly-same as following Python implementation:

@classmethod
def _backward(cls, root: TensorPyBackend) -> None:
    ...
    init_grad = dispatch(Ops.ones, to_backend=root.backend)(shape=[1])
    grad = TensorPyBackend(data=init_grad, backend=root.backend)
    
    stack = [(grad, root.grad_fn)]
    while stack:
        grad, fn = stack.pop()  # fn: node, grad: incoming grad to node
        # compute grad output of current node 
        grads = fn.backward(grad)
        # for each grad output and current node's parent tensors
        for grad, parent_tensor in zip(grads, fn._parents):
            # parent is non-leaf tensor
            if parent_tensor.grad_fn:
                stack.append((grad, parent_tensor.grad_fn))
                assert not parent_tensor.requires_grad
            # parent is leaf tensor and also 'is-param'
            if parent_tensor.requires_grad:
                if parent_tensor.grad is None:  # if grad not initialized
                    zeros_op = dispatch(Ops.zeros_like, to_backend=grad.backend)
                    parent_tensor.grad = TensorPyBackend(
                        zeros_op(parent_tensor.data), backend=grad.backend
                    )
                    grad_acc_op = dispatch(Ops.add_inplace, to_backend=grad.backend)
                    grad_acc_op(parent_tensor.grad.data, grad.data)
*/
void AutogradEngine::backward(
    const TensorPtr& root,
    const std::optional<TensorPtr>& gradient,
    const bool retain_graph
) {
    DEBUG("[Autograd] Backward process started with root" << root);
    TensorPtr grad;
    if (gradient && *gradient) {
        if ((*gradient)->shape_ != root->shape_) {
            throw std::runtime_error(
                "[Autograd] Initial incoming gradient shape mismatch with root"
            );
        }
        grad = *gradient;
    } else {
        grad = Tensor::create(root->shape_, 1.0f);
    }
    std::stack<std::pair<TensorPtr, FunctionPtr>> stack;
    stack.push({grad, root->grad_fn_});

    DEBUG("[Autograd] Root fn: " << root->grad_fn_);

    while (!stack.empty()) {
        auto [grad, fn] = stack.top(); stack.pop();
        DEBUG_ASSERT(fn);
        // compute grad output for each node
        // TODO: implement lazy evaluation
        TensorPtrVec grads = fn->backward(grad);
        DEBUG_SCOPE("Grad NaN check");
        {
            for (auto grad : grads) {
                grad->check_nan("Grad of: " + fn->name());
            }
        }
        // for each node's grad output
        for (size_t i = 0; i < grads.size(); i++) {
            // get match node's parent Tensor
            TensorPtr parent = fn->parents_[i];
            
            if (parent->grad_fn_) {
                // 1. the original parent was non-leaf tensor (has grad_fn)
                // 2. non-leaf's requires_grad must be always false
                DEBUG_ASSERT(!parent->requires_grad_);
                // Then push to DFS stack
                // TODO: implement lazy evaluation
                stack.push({grads[i], parent->grad_fn_});
            }
            if (parent->requires_grad_) {  // if leaf and 'is param',
                if (!parent->grad_)
                    // if grad not initialized, init zero Tensor
                    // if one uses optimizer, then always initialized.
                    parent->grad_ = Tensor::create(parent->shape_, 0.0f);
                // TODO: use dispatcher
                op::add_inplace_contiguous(parent->grad_, grads[i]);  // accumulate grad
                DEBUG("[Autograd] Accumulated grad to tensor: " << parent);
            }
        }
    }
    DEBUG("[Autograd] Backward process finished");
    AutogradEngine::clear_graph();
}
