#include "backward.hpp"


PYBIND11_MODULE(libcpp, m) {
    m.def("sigmoid_backward", &sigmoid_backward);
    m.def("mmul_left_backward", &mmul_left_backward);
    m.def("mmul_right_backward", &mmul_right_backward);
    m.def("add_broadcast_backward", &add_broadcast_backward);
    m.def("ce_softmax_mean_backward", &ce_softmax_mean_backward);
}
