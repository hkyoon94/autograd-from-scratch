#pragma once

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <stack>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>

#include "constants/enums.h"
#include "function/function.h"
#include "ops/cuda/utils.h"
#include "tensor/tensor.h"
#include "utils/utils.h"

namespace py = pybind11;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrVec = std::vector<TensorPtr>;


inline size_t calc_numel(const std::vector<size_t>& shape) {
    return std::accumulate(
        shape.begin(), 
        shape.end(), 
        (size_t)1, 
        std::multiplies<size_t>()
    );
}


// stride computation (row-major oriented)
inline std::vector<size_t> calc_contiguous_stride(const std::vector<size_t>& shape) {
    std::vector<size_t> stride(shape.size());
    size_t acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        stride[i] = acc;
        acc *= shape[i];
    }
    return stride;
}


struct DataPtrStorage {
    /*  Raw data pointers. 
        Only Tensor instances can access this. */
    void* _data_;
    size_t size_;
    DType dtype_;   // TODO: Expand to support multiple DTypes
    Device device_;
    std::optional<py::object> owner_ = std::nullopt;

    private:
        DataPtrStorage(
            float* data,
            size_t size,
            DType dtype = DType::FP32,
            Device device = Device::CPU,
            std::optional<py::object> owner = std::nullopt
        )
        : _data_(data), size_(size), dtype_(dtype), device_(device), owner_(owner) {}

    public:
        ~DataPtrStorage() {
            // _free_data();
        }

        // * Accessible constructor
        template <typename... Args>
        inline static std::shared_ptr<DataPtrStorage> create(Args&&... args) {
            struct make_shared_enabler : public DataPtrStorage {
                make_shared_enabler(Args&&... args)
                : DataPtrStorage(std::forward<Args>(args)...) {}
            };
            return std::make_shared<make_shared_enabler>(
                std::forward<Args>(args)...
            );
        }

        void _free_data() {
            if (!this->owner_) {
                if (this->device_ == Device::CPU)
                    free(this->_data_);
                else if (this->device_ == Device::CUDA)
                    cudaFree(this->_data_);
                DEBUG("Tensor data freed");
            }
        }
};


class Tensor : public std::enable_shared_from_this<Tensor> {
    using TensorPtr = std::shared_ptr<Tensor>;
    using FunctionPtr = std::shared_ptr<Function>;

    private:
        std::shared_ptr<DataPtrStorage> _stor_;

    public:
        std::vector<size_t> shape_;
        std::vector<size_t> stride_;
        size_t numel_;
        DType dtype_;
        Device device_;
        bool contiguous_;
        bool requires_grad_ = false;

        TensorPtr grad_{nullptr};
        FunctionPtr grad_fn_{nullptr};

    private:  // constructors  
        // * Constructors are privatized, only accesible through Tensor::create()
        Tensor(
            py::buffer& array,
            const std::vector<size_t>& shape,
            DType dtype = DType::FP32,
            Device device = Device::CPU
        );  // overload
        Tensor(
            const std::vector<size_t>& shape,
            float value,
            DType dtype = DType::FP32,
            Device device = Device::CPU
        );  // overload
        Tensor(
            const std::vector<size_t>& shape,
            DType dtype = DType::FP32,
            Device device = Device::CPU
        );  // overload
    
    public:
        ~Tensor();
        // copy ctor (Tensor b = a;)
        Tensor(const Tensor&) = default;
        // copy-assign ctor (b = a;)
        Tensor& operator=(const Tensor&) = default;
        // move ctor (Tensor b = std::move(a))
        Tensor(Tensor&& other) = default;
        // move-assign ctor (b = std::move(a))
        Tensor& operator=(Tensor&& other) = default;
    
    private:
        size_t _get_index(py::list slice, const std::vector<size_t>& stride) const;
        size_t _get_storage_idx(size_t tensor_idx) const;
    
    public:
        // * Accessible constructor
        template <typename... Args>
        inline static std::shared_ptr<Tensor> create(Args&&... args) {
            struct make_shared_enabler : public Tensor {
                make_shared_enabler(Args&&... args)
                : Tensor(std::forward<Args>(args)...) {}
            };
            return std::make_shared<make_shared_enabler>(
                std::forward<Args>(args)...
            );
        }

        // * Retrieving raw data-ptr in cuda-runtime format
        template <typename DataType>
        inline const DataType* data() const {
            return reinterpret_cast<const DataType*>(this->_stor_->_data_);
        }
        
        template <typename DataType>  // read-write overload
        inline DataType* data() {
            return reinterpret_cast<DataType*>(this->_stor_->_data_);
        }

        // * Retrieving raw data-ptr in cuda-driver API format
        template <typename DataType>
        inline CUdeviceptr data_cu() const {
            return reinterpret_cast<CUdeviceptr>(this->data<DataType>());
        }

        TensorPtr to(Device device);
        bool is_contiguous() const;
        TensorPtr broadcast_to(const std::vector<size_t>& out_shape) const;
        // TensorPtr slice(const std::vector<size_t>& slice);
        TensorPtr contiguous();
        void backward(const std::optional<TensorPtr> init_grad = std::nullopt);
        py::array numpy();
        std::optional<float> getelement(py::list slice) const;
        uintptr_t id() const;
        std::string __repr__() const;

        // For NaN debug
        inline void check_nan(const std::string& tag) const {
            const float* data__ = this->data<const float>();
            for (size_t i = 0; i < numel_; ++i) {
                if (!std::isfinite(data__[i])) {
                    std::cerr << "[NaN] " << tag
                            << " @ idx " << i
                            << " val " << data__[i] << std::endl;
                    std::abort();
                }
            }
        }

        void _replace_ptr(void* new_ptr);
};


inline std::ostream& operator<<(std::ostream& os, const TensorPtr& t) {
    return os << t->__repr__();
}
