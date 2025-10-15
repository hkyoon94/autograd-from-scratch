#include <stdexcept>
#include <cuda_runtime.h>
#include <string>

#include "allocator.h"
#include "core.h"
#include "functional.h"
#include "ops_c.h"
#include "ops_cuda.h"
#include "utils.h"

using namespace std;
namespace py = pybind11;

using TensorPtr = std::shared_ptr<Tensor>;
using FunctionPtr = std::shared_ptr<Function>;


// Tensor
// TODO: Create 'to' device function

// initializing from python array.
Tensor::Tensor(
    py::buffer& array,
    const std::vector<size_t>& shape,
    DType dtype,
    Device device
) 
: shape_(shape) {
    py::buffer_info info = array.request();
    this->numel_ = calc_numel(this->shape_);
    this->stride_ = calc_contiguous_stride(this->shape_);
    float* _buffer = static_cast<float*>(info.ptr);
    // TODO: Allow multiple-dtypes (data-ptr factory method?)
    float* data_ptr;
    if (device == Device::CPU) {
        data_ptr = _buffer;
    }
    else if (device == Device::CUDA) {
        size_t _size = this->numel_ * sizeof(float);
        CUDA_CHECK(cudaMalloc(&data_ptr, _size));
        CUDA_CHECK(cudaMemcpy(
            data_ptr, _buffer, _size, cudaMemcpyHostToDevice
        ));
    }
    this->_stor_ = DataPtrStorage::create(
        data_ptr,
        this->numel_,
        DType::FP32,
        device,
        array
    );
    this->contiguous_ = true;
    this->dtype_ = dtype;
    this->device_ = device;
    DEBUG(
        "Tensor init 'from array', with shape: " << shape <<
        ", <addr: " << this->_stor_ << ">, owns: " << false <<
        ", <device: " << this->device_ << ">"
    );
}

// initializing as specific value
Tensor::Tensor(
    const std::vector<size_t>& shape,
    float value,
    DType dtype,
    Device device
) : shape_(shape) {
    this->numel_ = calc_numel(this->shape_);
    this->stride_ = calc_contiguous_stride(this->shape_);
    // TODO: Allow multiple-dtypes (data-ptr factory method?)
    float* data_ptr;
    if (device == Device::CPU) {
        // data_ptr = static_cast<float*>(malloc(this->numel_ * sizeof(float)));
        data_ptr = static_cast<float*>(
            mallocator_temp.allocate(this->numel_ * sizeof(float))
        );
        std::fill(
            data_ptr,
            data_ptr + this->numel_,
            value
        );
    }
    else if (device == Device::CUDA) {
        size_t _size = this->numel_ * sizeof(float);
        CUDA_CHECK(cudaMalloc(&data_ptr, _size));
        CUDA_CHECK(cudaMemset(data_ptr, value, _size));
    }
    this->_stor_ = DataPtrStorage::create(
        data_ptr,
        numel_,
        DType::FP32,
        device,
        std::nullopt
    );
    this->contiguous_ = true;
    this->dtype_ = dtype;
    this->device_ = device;
    DEBUG(
        "Tensor init 'temp-malloc+value', with shape: " << shape << ", value: " << value <<
        ", <addr: " << this->_stor_ << ">, owns: " << true <<
        ", <device: " << this->device_ << ">"
    );
}

// initializing only malloc
Tensor::Tensor(
    const std::vector<size_t>& shape,
    DType dtype,
    Device device
) : shape_(shape) {
    this->numel_ = calc_numel(this->shape_);
    this->stride_ = calc_contiguous_stride(this->shape_);
    // TODO: Allow multiple-dtypes (data-ptr factory method?)
    float* data_ptr;
    if (device == Device::CPU) {
        // data_ptr = static_cast<float*>(malloc(this->numel_ * sizeof(float)));
        data_ptr = static_cast<float*>(
            mallocator_temp.allocate(this->numel_ * sizeof(float))
        );
    }
    else if (device == Device::CUDA) {
        cudaMalloc(&data_ptr, this->numel_ * sizeof(float));
    }
    this->_stor_ = DataPtrStorage::create(
        data_ptr,
        this->numel_,
        DType::FP32,
        device,
        std::nullopt
    );
    this->contiguous_ = true;
    this->dtype_ = dtype;
    this->device_ = device;
    DEBUG(
        "Tensor init 'only-temp-malloc', with shape: " << shape <<
        ", <addr: " << this->_stor_ << ">, owns: " << true <<
        ", <device: " << this->device_ << ">"
    );
}

Tensor::~Tensor() {}

void Tensor::_replace_ptr(void* new_ptr) {
    this->_stor_->_data_ = static_cast<float*>(new_ptr);
}

bool Tensor::is_contiguous() const {
    size_t ndim = this->shape_.size();
    size_t expected_stride = 1;
    for (int d = ndim - 1; d >= 0; --d) {
        if (shape_[d] == 0)
            return true;
        if (stride_[d] != expected_stride)
            return false;
        expected_stride *= shape_[d];
    }
    return true;
}

TensorPtr Tensor::contiguous() {
    if (this->contiguous_) {
        return shared_from_this();
    }
    TensorPtr out = Tensor::create(this->shape_);
    auto out_stride_ = calc_contiguous_stride(this->shape_);
    float* out_ptr = out->data<float>();
    const float* ptr = this->data<const float>();
    for (size_t tidx = 0; tidx < this->numel_; ++tidx) {
        size_t sidx = _get_storage_idx(tidx);
        out_ptr[tidx] = ptr[sidx];
    }
    return out;
}

inline size_t Tensor::_get_storage_idx(size_t tensor_idx) const {
    size_t ndim = this->shape_.size();
    size_t storage_idx = 0;
    for (size_t dim = 0; dim < ndim; ++dim) {
        size_t i = tensor_idx % shape_[ndim - 1 - dim];
        tensor_idx /= shape_[ndim - 1 - dim];
        storage_idx += i * stride_[ndim - 1 - dim];
    }
    return storage_idx;
}

TensorPtr Tensor::broadcast_to(const std::vector<size_t>& out_shape) const {
    Tensor y = *this;  // shallow copy
    std::vector<size_t> new_stride(out_shape.size());
    int ndim_diff = out_shape.size() - this->shape_.size();

    for (size_t i = 0; i < out_shape.size(); i++) {
        size_t dim = (i < ndim_diff) ? 1 : shape_[i - ndim_diff];
        if (dim == out_shape[i])
            new_stride[i] = (i < ndim_diff) ? 0 : stride_[i - ndim_diff];
        else if (dim == 1)
            new_stride[i] = 0;  // broadcast
        else
            throw std::runtime_error("incompatible shape in broadcast_view");
    }
    y.shape_ = out_shape;
    y.stride_ = new_stride;
    y.numel_ = calc_numel(out_shape);
    y.contiguous_ = false;
    return std::make_shared<Tensor>(y);
}

TensorPtr Tensor::to(Device device) {    
    cudaMemcpyKind kind;
    if (device_ == Device::CPU && device == Device::CUDA)
        kind = cudaMemcpyHostToDevice;
    else if (device_ == Device::CUDA && device == Device::CPU)
        kind = cudaMemcpyDeviceToHost;
    else if (device_ == Device::CUDA && device == Device::CUDA)
        kind = cudaMemcpyDeviceToDevice;
    else
        return shared_from_this();

    TensorPtr _new = Tensor::create(this->shape_, this->dtype_, device);
    assert(this->numel_ == _new->numel_);
    cudaMemcpy(
        _new->data<float>(),
        this->data<const float>(), 
        this->numel_ * sizeof(float),
        kind
    );
    DEBUG("Copied data '" << this->device_ << "' -> '" << device << "'");
    return _new;
}

void Tensor::backward(const optional<TensorPtr> init_grad) {
    AutogradEngine::backward(shared_from_this(), init_grad);
}

inline size_t Tensor::_get_index(
    py::list slice, const std::vector<size_t>& stride
) const {
    size_t idx = 0;
    for (size_t i = 0; i < stride.size(); i++)
        idx += slice[i].cast<size_t>() * stride[i];
    DEBUG("idx: " << idx);
    return idx;
}

std::string Tensor::__repr__() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); i++) {
        oss << shape_[i];
        if (i + 1 < shape_.size()) oss << ", ";
    }
    auto ptr = data<const float>();
    oss << "], data=[";
    for (size_t i = 0; i < std::min<size_t>(10, numel_); i++) {
        oss << ptr[i];
        if (i + 1 < std::min<size_t>(10, numel_)) oss << ", ";
    }
    if (numel_ > 10) oss << "...";
    oss << "])";
    return oss.str();
}

std::optional<float> Tensor::getelement(py::list slice) const {
    size_t idx = _get_index(slice, stride_);
    if ((0 <= idx) && (idx < numel_))
        return data<const float>()[idx];
    else
        throw std::runtime_error("Index out of range");
}

uintptr_t Tensor::id() const {
    return reinterpret_cast<uintptr_t>(&this->_stor_->_data_);
}

py::array Tensor::numpy() {
    std::vector<ssize_t> shape(shape_.begin(), shape_.end());
    std::vector<ssize_t> strides(shape.size());
    ssize_t stride = sizeof(float);
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return py::array(
        py::buffer_info(
            _stor_->_data_,
            sizeof(float),
            py::format_descriptor<float>::format(),
            shape.size(),
            shape,
            strides
        )
    );
}


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
    AutogradEngine::graph_.edges.clear();
}
