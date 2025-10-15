#include <cstdlib>

// #include <numpy/arrayobject.h>  // for build test
// #include <torch/torch.h>  // for build test
#include <cuda_runtime.h>  // for build test
#include <cuda.h>
#include <memory>

#include "allocator.h"
#include "core.h"
#include "functional.h"
#include "ops_c.h"
#include "optimizer.h"
// #include "ops_cuda.h"


int main() {
    AutogradEngine::on(false);
    AutogradEngine::on(true);

    /* Brief test script for testing forward -> backward, via C-backend Autograd API */
    // Initialize Tensor Ptrs
    TensorPtr x1 = op::_zeros({10, 23});
    TensorPtr w1 = op::_zeros({23, 53});
    TensorPtr w2 = op::_zeros({10, 53});
    
    w1->requires_grad_ = true;
    w2->requires_grad_ = true;

    // Initialize functions
    Sum sum_fn = Sum();
    MatMul mm_fn = MatMul();
    View view_fn = View();

    // Proceed chained forward
    TensorPtr x2 = mm_fn.forward({x1, w1});
    x2 = view_fn.forward({x2}, vector<size_t>{53, 10});
    TensorPtr x3 = mm_fn.forward({x2, w2});
    TensorPtr x4 = sum_fn.forward({x3}, {0, 1});
    
    // std::cout << AutogradEngine::graph_.edges << std::endl;
    // Calls autograd
    x4->backward();

    std::cout << "w1 grad: " << w1->grad_ << std::endl;
    std::cout << "w2 grad: " << w2->grad_ << std::endl;


    /* Brief testing optimizer & memory pool */
    TensorPtr p = Tensor::create(std::vector<size_t>{2, 2}, 0.0);
    p->requires_grad_ = true;
    std::cout << "p (before update): " << p << std::endl;

    std::cout << "Temp pool used: " << mallocator_temp.used() << std::endl;
    std::cout << "Persistent pool used: " << mallocator_pers.used() << std::endl;

    std::shared_ptr<SGDOptimizer> sgd = SGDOptimizer::create(TensorPtrVec{p}, 0.1);

    std::cout << "Temp pool used: " << mallocator_temp.used() << std::endl;
    std::cout << "Persistent pool used: " << mallocator_pers.used() << std::endl;

    p->grad_ = Tensor::create(p->shape_, 1.0);
    sgd->step();

    std::cout << "p (after update): " << p << std::endl;
    sgd->zero_grad();

    std::cout << "p.grad (after zero_grad) " << p->grad_ << std::endl;
    std::cout << "p (after zero_grad): " << p << std::endl;

    std::cout << "Temp pool used: " << mallocator_temp.used() << std::endl;
    std::cout << "Persistent pool used: " << mallocator_pers.used() << std::endl;


    // TODO: currently ptx kernel integration is under development
    // /* Brief test script for testing ptx kernel integration */
    // // Testing 'cuop::add'
    // // Initializing test tensors
    // size_t N = 4096; 
    // TensorPtr x = Tensor::create(vector{N}, 1.0f);
    // TensorPtr y = Tensor::create(vector{N}, 1.0f);
    // std::cout << x << std::endl;
    // std::cout << y << std::endl;

    // // Copying Tensors to device memory
    // x = x->to(Device::CUDA);
    // y = y->to(Device::CUDA);

    // // PTX kernel execution
    // TensorPtr z = cuop::add(x, y)->to(Device::CPU);
    // std::cout << op::sum(z, {0}) << std::endl;


    // /* Brief test script for testing ptx kernel integration */
    // // Initializing test tensors
    // // Testing 'cuop::mm'
    // size_t M = 1000;
    // N = 500;
    // size_t K = 750;

    // x = Tensor::create(vector{M, K}, 1.0f);
    // y = Tensor::create(vector{K, N}, 1.0f);
    // std::cout << x << std::endl;
    // std::cout << y << std::endl;
    // x = x->to(Device::CUDA);
    // y = y->to(Device::CUDA);

    // z = cuop::mm(x, y);
    // // // TensorPtr z = cuop::mm2(x, y, "naive");

    // z = z->to(Device::CPU);
    // std::cout << op::sum(z, {0, 1}) << std::endl;

    return 0;
}
