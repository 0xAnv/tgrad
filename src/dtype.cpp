//
// Created by anvesh on 4/2/26.
//

#include "dtype.h"
#include <stdexcept>

namespace tgrad {
    size_t dtype_size(const DType dtype) {
        switch (dtype) {
        case DType::Float32: return 4; // 32 bts = 4 bytes
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
        case DType::Bool: return 1;
        default: throw std::runtime_error("Unknow type");
        }
    }

    // returns nice string for printing and debugging
    std::string dtype_name(const DType dtype) {
        switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::Bool: return "bool";
        default: return "Unknown dtype";
        }
    }
}
