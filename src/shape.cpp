//
// Created by anvesh on 4/3/26.
//

#include "shape.h"

#include <numeric> // std::accumulate
#include <functional> // std::multiplies

namespace tgrad {
    Strides compute_contiguous_strides(const Shape& shape) {
        /*
         * This computes contiguous strides given shape
         * Example: shape[2,3,4] -> strides[12,4,1]
         */
        Strides strides(shape.size());
        SizeType current_stride = 1;
        const int shape_size = static_cast<int>(shape.size() - 1); // ulong -> int
        for (int i = shape_size; i >= 0; i--) {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }
        return strides;
    }

    SizeType shape_numel(const Shape& shape) {
        /* Count total number of elements in shape */
        if (shape.empty()) return 0;
        // std::accumulate loops over vec ; starts with 1 and multiplies everything together
        return std::accumulate(shape.begin(), shape.end(),
                               1LL, std::multiplies<SizeType>());
    }

    bool shapes_are_broadcastable(const Shape& a, const Shape& b) {
        int i = static_cast<int>(a.size() - 1), j = static_cast<int>(b.size() - 1); // ptr to end
        while (i >= 0 && j >= 0) {
            // maths does not align
            if (a[i] != b[j] && a[i] != 1 && b[j] != 1) return false;
            i--, j--;
        }
        return true;
    }

    Shape broadcast_shape(const Shape& a, const Shape& b) {
        Shape result(std::max(a.size(), b.size()));

        int i = static_cast<int>(a.size() - 1),
            j = static_cast<int>(b.size() - 1),
            k = static_cast<int>(result.size() - 1);

        while (k >= 0) {
            // if shape ran out of dims, we pretend its 1
            SizeType dim_a = (i >= 0) ? a[i] : 1;
            SizeType dim_b = (j >= 0) ? b[j] : 1;

            result[k] = std::max(dim_a, dim_a);
            i--;
            j--;
            k--;
        }
        return result;
    }

    bool is_contiguous(const Shape& shape, const Strides& strides) {
        // we check if mem has gaps: calc perfect strides and compare to ours
        const Strides expected = compute_contiguous_strides(shape);
        return expected == strides;
    }
}

