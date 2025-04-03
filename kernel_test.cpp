#include <iostream>

#include "backward.hpp"


int main() {
    torch::Tensor x1 = torch::randn({128, 256});
    torch::Tensor x2 = torch::randn({128, 64});
    torch::Tensor x3 = torch::randn({32, 256});
    torch::Tensor h0 = torch::randn({128, 256});
    torch::Tensor x3t = x3.t();

    // std::cout << "x1:" << std:: endl << x1 << std::endl;
    // std::cout << "x2:" << std:: endl << x2 << std::endl;
    // std::cout << "x3:" << std:: endl << x3 << std::endl;
    
    torch::Tensor z = sigmoid_backward(x1, x1);
    std::cout << "sigmoid_backward(x1, x1):" << std:: endl << z
    << " Shape: " << z.size(0) << " " << z.size(1) << std::endl;

    z = mmul_left_backward(x1, x3);
    std::cout << "mmul_left_backward(x1, x3):" << std:: endl << z
    << " Shape: " << z.size(0) << " " << z.size(1) << std::endl;

    z = mmul_right_backward(x1, x2);
    std::cout << "mmul_right_backward(x1, x2):" << std:: endl << z
    << " Shape: " << z.size(0) << " " << z.size(1) << std::endl;

    z = add_broadcast_backward(x1);
    std::cout << "add_broadcast_backward(x1):" << std:: endl << z
    << " Shape: " << z.size(0) << std::endl;

    z = ce_softmax_mean_backward(x3, h0, 3);
    std::cout << "ce_softmax_mean_backward(x1):" << std:: endl << z
    << " Shape: " << z.size(0) << " " << z.size(1) << std::endl;
    
    std::cout << "Test fin" << std::endl;
    return 0;
}
