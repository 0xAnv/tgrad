//
// Created by anvesh on 4/3/26.
//

#pragma once

#ifndef TGRAD_TENSOR_H
#define TGRAD_TENSOR_H

#include "device.h"
#include "dtype.h"
#include "storage.h"
#include "shape.h"

#include <vector>

namespace tgrad {
    struct GradFn;

    class Tensor {
    public:
        // default constructor; enables Tensor a;
        Tensor() = default;

        // Static factories of constructors
        static Tensor from_data(const std::vector<float>& data, Shape shape, bool requires_grad = false);
        static Tensor zeros(const Shape& shape, DType dtype = DType::Float32, Device device = Device::cpu());
        static Tensor ones(const Shape& shape, DType dtype = DType::Float32, Device device = Device::cpu());
        static Tensor randn(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());

        // getters
        const Shape& shape() const { return shape_; }
        const Strides& strides() const { return strides_; }
        SizeType ndim() const { return shape_.size(); }
        SizeType numel() const { return shape_numel(shape_); }
        DType dtype() const { return storage_->dtype(); }
        Device device() const { return storage_->device(); }
        const void* data_ptr() const { return storage_->data_ptr(); }
        void* data_ptr() { return storage_->data_ptr(); }

        // autograd
        bool requires_grad() const { return requires_grad_; }
        void set_requires_grad(const bool val) { requires_grad_ = val; }
        std::shared_ptr<Tensor> grad; // accum grad ( sets at .backward() )
        std::shared_ptr<GradFn> grad_fn; // func that created this tensor
        void backward() const; // run backward pass from this tensor to calc grads

        // Shape operations
        Tensor view(const Shape& new_shape) const; // fast; no copy reshape
        Tensor reshape(const Shape& new_shape) const; // slow; copies mem
        Tensor transpose(int dim_0, int dim_1) const;
        Tensor contiguous() const; // forces copy to flat mem
        Tensor T() const { return transpose(ndim() - 2, ndim() - 1); } // pytorch style T()
        Tensor matmul(const Tensor& other) const; // DOT product/ mat mul
        Tensor sum() const; // sums all elems to single scalar
        Tensor operator+(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const; // Element wise mul

        bool is_contiguous() const;
        static void print(); // Tensor p = {2,3}; t.print();

    private:
        std::shared_ptr<Storage> storage_;
        Shape shape_;
        Strides strides_;
        size_t storage_offset_ = 0;
        bool requires_grad_ = false;
        // private construcor for our maths funcs
        Tensor(std::shared_ptr<Storage> storage, Shape shape, Strides strides, size_t storage_offset = 0);
    };

    // non member matmul ops
    Tensor matmul(const Tensor& a, const Tensor& b);
}

#endif //TGRAD_TENSOR_H
