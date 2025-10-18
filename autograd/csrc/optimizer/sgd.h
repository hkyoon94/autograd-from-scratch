#pragma once

#include <alloca.h>
#include <memory>
#include <stdexcept>

#include "allocator/pool.h"
#include "tensor/tensor.h"
#include "ops/cpu/ops.h"
#include "utils/utils.h"


constexpr size_t PARALLEL_THRESHOLD = 16;


template <typename Derived>
class Optimizer {

    protected:
        Optimizer(TensorPtrVec params, float lr)
        : lr_(lr) {
            size_t num_no_params = 0;
            for (TensorPtr p : params) {
                if (!p->requires_grad_) {
                    num_no_params += 1;
                    continue;
                }
                /* Copying the param tensor's initial data to persistent memory pool */
                // TODO: various DType support is under development
                size_t bytes = p->numel_ * sizeof(float);
                void* persistent_ptr = mallocator_pers.allocate(bytes);
                std::memcpy(persistent_ptr, p->data<const float>(), bytes);
                DEBUG("Copied param " << p->shape_ << " data to persistent pool");
                p->_replace_ptr(persistent_ptr);
                p->grad_ = Tensor::create(p->shape_, 0.0f);
                this->params_.push_back(p);
            }
            this->num_param_tensors_ = this->params_.size();
            if (num_no_params)
                std::cout << "[Note] Detected " << num_no_params << \
                    " non-param tensors. These tensors will not be updated. " << std::endl;
            if (!this->num_param_tensors_) {
                throw std::runtime_error("No param-tensors detected.");
            }
            DEBUG(
                "Initialized " << this->_name_ << " with " 
                << this->num_param_tensors_ << " params"
            );
        }

    public:
        std::string _name_;
        TensorPtrVec params_;
        size_t num_param_tensors_;
        float lr_;

        virtual ~Optimizer() noexcept = default;

        // accessible constructor
        // This method should not be used from the base class 'Optimizer' itself.
        template <typename... Args>
        inline static std::shared_ptr<Derived> create(Args&&... args) {
            struct make_shared_enabler : public Derived {
                make_shared_enabler(Args&&... args)
                : Derived(std::forward<Args>(args)...) {}
            };
            return std::make_shared<make_shared_enabler>(
                std::forward<Args>(args)...
            );
        }

        void zero_grad() noexcept {
            #pragma omp parallel for if (this->num_param_tensors_ > PARALLEL_THRESHOLD) \
                schedule(static, 8)
            for (size_t i = 0; i < this->num_param_tensors_; ++i) {
                DEBUG("Calling optimizer zero_grad");
                std::fill_n(
                    params_[i]->grad_->data<float>(),
                    params_[i]->numel_,
                    0.0f
                );
            }
        }

        virtual void step() = 0;
};


class SGDOptimizer : public Optimizer<SGDOptimizer> {
    protected:
        SGDOptimizer(TensorPtrVec params_, float lr)
        : Optimizer<SGDOptimizer>(params_, lr) {}
    
    public:
        inline static constexpr const char* _name_ = "SGDOptimizer";

        ~SGDOptimizer() noexcept override = default; 

        void step() noexcept override {
            DEBUG("Calling optimizer step");
            #pragma omp parallel for if (this->num_param_tensors_ > PARALLEL_THRESHOLD) \
                schedule(static, 8)
            for (size_t i = 0; i < this->num_param_tensors_; ++i) {
                TensorPtr p = this->params_[i];
                // p->grad_->check_nan("p");
                op::add_inplace_contiguous(p, p->grad_, -this->lr_);
            }

            mallocator_temp.reset();  // TODO Make this as implicit macro?
        }
};
