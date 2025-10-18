#include <cstdlib>

#include <memory.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dispatcher/dispatcher.h"
#include "tensor/tensor.h"

using str = std::string;
using TensorPtr = std::shared_ptr<Tensor>;
using IArgs = std::vector<IValue>;
using IKwargs = std::unordered_map<std::string, IValue>;


// TODO: in development
struct Dispatcher {
    using ForwardFn = std::function<TensorPtr(const IArgs&, const IKwargs&)>;
    using BackwardFn = std::function<std::vector<TensorPtr>(const TensorPtr&)>;

    static Dispatcher& instance() {
        static Dispatcher inst;
        return inst;
    }

    void register_op(const str& name, ForwardFn fwd, BackwardFn bwd) {
        forwards_[name] = fwd;
        backwards_[name] = bwd;
    }

    TensorPtr call_forward(const str& name, const IArgs& args, const IKwargs& kwargs) {
        return forwards_.at(name)(args, kwargs);
    }

    std::vector<TensorPtr> call_backward(const str& name, const TensorPtr& grad) {
        return backwards_.at(name)(grad);
    }

private:
    std::unordered_map<str, ForwardFn> forwards_;
    std::unordered_map<str, BackwardFn> backwards_;
};
