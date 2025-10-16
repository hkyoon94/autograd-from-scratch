#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "functional.h"

class Tensor;


using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrVec = std::vector<TensorPtr>;


class Function {  // Virtual; i.e. abc
    protected:
        Function() {}

    public:
        const std::string _name_;
        virtual ~Function() = default;
        TensorPtrVec parents_;
        TensorPtrVec ctx_;
        std::vector<size_t> original_dim_;

        virtual std::string name() const = 0;

        // virtual TensorPtr forward(...) = 0;
        virtual TensorPtrVec backward(const TensorPtr& incoming_gradients) const = 0;

        template <typename FunctionClass>
        inline static std::shared_ptr<FunctionClass> create() {
            return std::make_shared<FunctionClass>();
        }

        uintptr_t id() const {
            return reinterpret_cast<uintptr_t>(this);
        }
};


class View : public Function {
    public:
        const std::string _name_ = "view";
        // TODO: Change return of forward to TensorPtrVec, to support multi-tensor output op
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs, const std::vector<size_t>& as_shape);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};

class Div : public Function {
    public:
        const std::string _name_ = "div";
        // multi-tensor output op
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs, const std::vector<float>& divisor);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};

class Sum : public Function {
    public:
        const std::string _name_ = "sum";
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs, const std::vector<size_t>& dims);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};

class Add : public Function {
    public:
        const std::string _name_ = "add";
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};

class MatMul : public Function {
    public:
        const std::string _name_ = "matmul";
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};

class Sigmoid : public Function {
    public:
        const std::string _name_ = "sigmoid";
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};

class LeakyRelu : public Function {
    public:
        const std::string _name_ = "leaky_relu";
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};

class SoftmaxCrossEntropyMean : public Function {
    public:
        const std::string _name_ = "softmax_ce_mean";
        std:: string name() const override;
        TensorPtr forward(const TensorPtrVec& inputs);
        TensorPtrVec backward(const TensorPtr& grad) const override;
};
