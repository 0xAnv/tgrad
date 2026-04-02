//
// Created by anvesh on 4/2/26.
//
#pragma once

#include <cstdint>
#include <string>

#ifndef TGRAD_DTYPE_H
#define TGRAD_DTYPE_H

namespace tgrad
{
    enum class DType : uint8_t
    {
        Float32,    // standard 32 bit format
        Float16,    // 16 bit format (half precision)
        BFloat16,   // Google brain float
        Int32,      // 32 bit int
        Int64,      // 64 bit int
        Bool        // Bool -> 1 byte
    };

    // debug funcs
    size_t dtype_size(DType dtype);
    std::string dtype_name(DType dtype);

}

#endif //TGRAD_DTYPE_H
