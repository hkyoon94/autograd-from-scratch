#pragma once

#include <iostream>


enum class DType { FP32, FP64, INT64, BOOL };
enum class Device { CPU, CUDA };


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
