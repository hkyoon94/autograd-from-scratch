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

#include "utils.h"


using namespace std;
namespace py = pybind11;


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


enum class DType { FP32, FP64, INT64, BOOL };
enum class Device { CPU, CUDA };


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
                DEBUG("Tensor data freed: <addr: " << this << ">");
            }
        }
};


class Function;

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
        void backward(const std::optional<TensorPtr> init_grad = nullopt);
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

using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrVec = std::vector<TensorPtr>;
using FunctionPtr = std::shared_ptr<Function>;

struct ComputationalGraph {
    using DirectedEdge = std::tuple<
        std::string,  // name of parent fn
        std::string,  // name of child fn
        std::vector<size_t>  // shape of intermediate tensor
    >;

    // directed edge: edges_[0] -> edges_[1]
    Counter<std::string> counter = Counter<std::string>{};
    std::vector<DirectedEdge> edges = std::vector<DirectedEdge>{};

    void add_edges(const TensorPtrVec& inputs, const FunctionPtr node);
};


class AutogradEngine {
    public:
        static bool on_;
        static bool track_graph_;
        static ComputationalGraph graph_;

        static void on(bool flag);  // AutogradEngine::on setter
        static void track_graph(bool flag);  // AutogradEngine::track_graph setter
        static ComputationalGraph get_graph();
        static void clear_graph();

        static void backward(
            const TensorPtr& root,
            const std::optional<TensorPtr>& gradient = std::nullopt,
            const bool retain_graph = false
        );
};


inline std::ostream& operator<<(std::ostream& os, const TensorPtr& t) {
    return os << t->__repr__();
}

inline std::ostream& operator<<(std::ostream& os, DType dt) {
    switch (dt) {
        case DType::FP32: os << "FP32"; break;
        case DType::FP64: os << "FP64"; break;
        case DType::INT64: os << "INT64"; break;
        case DType::BOOL: os << "BOOL"; break;
        default: os << "UnknownDType"; break;
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, Device dev) {
    switch (dev) {
        case Device::CPU: os << "CPU"; break;
        case Device::CUDA: os << "CUDA"; break;
        default: os << "UnknownDevice"; break;
    }
    return os;
}
