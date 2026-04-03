//
// Created by anvesh on 4/3/26.
//
#pragma once

#ifndef TGRAD_SHAPE_H
#define TGRAD_SHAPE_H

#include <cstdint>
#include <vector>

namespace tgrad {
    // type aliasing
    using SizeType = int64_t;
    using Shape = std::vector<SizeType>;
    using Strides = std::vector<SizeType>;

    // core maths functions for shapes
    SizeType shape_numel(const Shape& shape);
    Strides compute_contiguous_strides(const Shape& shape);
    // broadcasting funcs when shape do not match
    bool shapes_are_broadcastable(const Shape& a, const Shape& b);
    Shape broadcast_shape(const Shape& a, const Shape& b);
    // Memory check
    bool is_contiguous(const Shape& shape, const Strides& strides);
}

#endif //TGRAD_SHAPE_H
