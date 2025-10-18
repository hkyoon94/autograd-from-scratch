#include "allocator/pool.h"
#include "engine/backward.h"
#include "ops/cuda/utils.h"
#include "tensor/tensor.h"
#include "utils/utils.h"

using TensorPtr = std::shared_ptr<Tensor>;
using FunctionPtr = std::shared_ptr<Function>;


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

void Tensor::backward(const std::optional<TensorPtr> init_grad) {
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
