//
// Created by anvesh on 4/3/26.
//

/*
 *
 * TESTING TENSOR HERE
 *
 *
 */
#include "tensor.h"

#include <cassert>
#include <iostream>

using namespace tgrad;

int main() {
    std::cout << "------ Testing tgrad::Tensor ------" << std::endl;

    // test creation and shape maths
    const Tensor t = Tensor::zeros({2, 3});
    assert(t.ndim() == 2);
    assert(t.numel() == 6);
    assert(t.is_contiguous() == true);
    std::cout << "[PASS] Tensor creation and shape maths" << std::endl;

    // Test view() ; no mem cpy
    const Tensor v = t.view({6});
    assert(v.ndim() == 1);
    assert(v.numel() == 6);
    assert(v.is_contiguous() == true);
    assert(v.data_ptr() == t.data_ptr());
    std::cout << "[PASS] Tensor view accessors & mem sharing" << std::endl;

    // Test transpose()
    const Tensor tr = t.transpose(0, 1);
    assert(tr.shape()[0] == 3);
    assert(tr.shape()[1] == 2);
    assert(tr.is_contiguous() == false);    std::cout << "[PASS] Tensor transpose" << std::endl;

    std::cout << "ALL TESTS PASSED" << std::endl;
}