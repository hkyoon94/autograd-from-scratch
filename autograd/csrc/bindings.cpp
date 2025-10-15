#include <cstddef>

#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "allocator.h"
#include "core.h"
// #include "dispatch.h"  // TODO: currently under development
#include "functional.h"
#include "ops_c.h"
#include "ops_common.h"
// #include "ops_cuda.h"  // TODO: currently under development
#include "optimizer.h"

namespace py = pybind11;


PYBIND11_MODULE(_C, m) {

    // Tensor
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(
            py::init([](const std::vector<size_t>& shape) {
                return Tensor::create(shape);
            }),
            py::arg("shape")
        )
        .def(
            py::init([](const std::vector<size_t>& shape, float value) {
                return Tensor::create(shape, value);
            }),
            py::arg("shape"),
            py::arg("value")
        )
        .def(
            py::init([](py::buffer buf, const std::vector<size_t>& shape) {
                return Tensor::create(buf, shape);
            }),
            py::arg("buffer"),
            py::arg("shape")
        )
        .def("to", &Tensor::to)
        .def("contiguous", &Tensor::contiguous)
        .def("broadcast_to", &Tensor::broadcast_to)
        .def("backward", &Tensor::backward, py::arg("init_grad") = py::none())
        .def("numpy", &Tensor::numpy)
        .def("getelement", &Tensor::getelement)
        .def("check_nan", &Tensor::check_nan)
        .def("__repr__", &Tensor::__repr__)
        // necessary attributes
        .def_property_readonly(
            "shape", [](const Tensor &self) {return self.shape_;}
        )
        .def_property_readonly(
            "stride", [](const Tensor &self) {return self.stride_;}
        )
        .def_property_readonly(
            "numel", [](const Tensor &self) {return self.numel_;}
        )
        .def_property_readonly(
            "is_contiguous", [](const Tensor &self) {return self.contiguous_;}
        )
        .def_readwrite("grad", &Tensor::grad_)
        .def_readwrite("grad_fn", &Tensor::grad_fn_)
        .def_readwrite("requires_grad", &Tensor::requires_grad_);

    
    // Enums
    py::enum_<DType>(m, "DType")
        .value("fp32", DType::FP32)
        .value("fp64", DType::FP64)
        .value("int64", DType::INT64)
        .value("bool", DType::BOOL)
        .export_values();

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();
    

    // Functions
    py::class_<Function, std::shared_ptr<Function>>(m, "Function");

    py::class_<View, Function, std::shared_ptr<View>>(m, "View")
        .def(py::init<>())
        .def("forward", &View::forward);

    py::class_<Div, Function, std::shared_ptr<Div>>(m, "Div")
        .def(py::init<>())
        .def("forward", &Div::forward);

    py::class_<Sum, Function, std::shared_ptr<Sum>>(m, "Sum")
        .def(py::init<>())
        .def("forward", &Sum::forward);

    py::class_<Add, Function, std::shared_ptr<Add>>(m, "Add")
        .def(py::init<>())
        .def("forward", &Add::forward);

    py::class_<MatMul, Function, std::shared_ptr<MatMul>>(m, "MatMul")
        .def(py::init<>())
        .def("forward", &MatMul::forward);
    
    py::class_<Sigmoid, Function, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward);

    py::class_<LeakyRelu, Function, std::shared_ptr<LeakyRelu>>(m, "LeakyRelu")
        .def(py::init<>())
        .def("forward", &LeakyRelu::forward);
    
    py::class_<SoftmaxCrossEntropyMean, Function, std::shared_ptr<SoftmaxCrossEntropyMean>>(
        m, "SoftmaxCrossEntropyMean"
    )   .def(py::init<>())
        .def("forward", &SoftmaxCrossEntropyMean::forward);

    
    // Computational Graph
    py::class_<ComputationalGraph>(m, "ComputationalGraph")
        .def_property_readonly("edges", [](const ComputationalGraph& self) { return self.edges; });


    // Autograd
    py::class_<AutogradEngine>(m, "AutogradEngine")
        .def_static("on", &AutogradEngine::on)
        .def_static("track_graph", &AutogradEngine::track_graph)
        .def_static("get_graph", &AutogradEngine::get_graph);


    // Optimizers
    py::class_<Optimizer<SGDOptimizer>, \
        std::shared_ptr<Optimizer<SGDOptimizer>>>(m, "BaseOptimizer");
        // TODO: why need above?
    py::class_<SGDOptimizer, Optimizer<SGDOptimizer>, std::shared_ptr<SGDOptimizer>>(
        m, "SGDOptimizer"
    )
        .def(
            py::init([](const TensorPtrVec& params, float lr) {
                    return SGDOptimizer::create(params, lr);
                }
            ),
            py::arg("params"),
            py::arg("lr") = 1e-3f
        )
        .def("step", &SGDOptimizer::step)
        .def("zero_grad", &SGDOptimizer::zero_grad)
        .def_readwrite("lr", &SGDOptimizer::lr_)
        .def_readwrite("params", &SGDOptimizer::params_);


    // Memory
    // temp allocator reset
    m.def("_flush_temp", []() {
        mallocator_temp.reset();
    }, "Reset the temporary allocator");
    // persistent allocator reset
    m.def("_flush_persistent", []() {
        mallocator_pers.reset();
    }, "Reset the persistent allocator");


    // basic operations
    /* Initializers */
    m.def("zeros", &op::zeros);
    m.def("zeros_like", &op::zeros_like);
    m.def("ones", &op::ones);
    m.def("ones_like", &op::ones_like);
    m.def("randn", &op::randn);

    /* View Ops Common */
    m.def("_broadcast_shapes", &_broadcast_shapes);
    m.def("_dims_to_reduce", &_dims_to_reduce);
    m.def("view", &op::view);

    /* Operators */
    // sum
    m.def("sum", &op::sum);
    m.def("sum_backward", &op::sum_backward);
    // add
    m.def("add", &op::add);
    m.def("add_backward", &op::add_backward);
    m.def("add_inplace", &op::add_inplace_contiguous);
    // mm
    m.def("mm", &op::mm);
    m.def("mm_backward", &op::mm_backward);

    /* activations */
    // sigmoid
    m.def("sigmoid", &op::sigmoid);
    m.def("sigmoid_backward", &op::sigmoid_backward);
    // relu
    m.def("leaky_relu", op::leaky_relu);
    m.def("leaky_relu_backward", op::leaky_relu_backward);

    /* Nonlinear */
    m.def("ce_softmax_mean", op::ce_softmax_mean);
    m.def("ce_softmax_mean_backward", &op::ce_softmax_mean_backward);

    // m.def("cuadd", &cuop::add);
    // m.def("cumm", &cuop::mm);
    // m.def("cumm2", &cuop::mm2);
}
