//
// Created by anvesh on 4/3/26.
//

#include "tensor.h"

#include <utility>
#include <cstring>

namespace tgrad {
    // private constructor
    Tensor::Tensor(std::shared_ptr<Storage> storage, Shape shape, Strides strides, const size_t storage_offset)
        : storage_(std::move(storage)),
          shape_(std::move(shape)),
          strides_(std::move(strides)),
          storage_offset_(storage_offset) {}

    // static factories of constructors
    Tensor Tensor::from_data(const std::vector<float>& data, const Shape shape, const bool requires_grad) {
        // calc how many bytes we need
        const size_t num_bytes = data.size() * dtype_size(DType::Float32);
        // confirm shape mismatch isnt there
        if (static_cast<SizeType>(data.size()) != shape_numel(shape))
            throw
                std::runtime_error("Data size does not match shape total elements");
        // creating storage block
        const auto storage = Storage::from_data(data.data(), num_bytes, DType::Float32);
        // computing strides
        const Strides strides = compute_contiguous_strides(shape);
        // build and reurn tensor
        Tensor t(storage, shape, strides);
        t.set_requires_grad(requires_grad);
        return t;
    }

    Tensor Tensor::zeros(const Shape& shape, const DType dtype, const Device device) {
        const size_t num_bytes = shape_numel(shape) * dtype_size(dtype);
        const auto storage = Storage::allocate(num_bytes, device, dtype);
        // physically writing zeros to every bit in ram
        if (device.is_cpu()) std::memset(storage->data_ptr(), 0, num_bytes);
        else throw std::runtime_error("zeros() not implemented for non CPU device");

        const Strides strides = compute_contiguous_strides(shape);
        auto t = Tensor(storage, shape, strides);
        return t;
    }

    Tensor Tensor::ones(const Shape& shape, const DType dtype, const Device device) {
        const size_t num_bytes = shape_numel(shape) * dtype_size(dtype);
        const auto storage = Storage::allocate(num_bytes, device, dtype);
        if (device.is_cpu()) {
            if (dtype == DType::Float32) {
                float* ptr = storage->data_as<float>();
                std::fill(ptr, ptr + shape_numel(shape), 1.0f);
            }
            else throw std::runtime_error("ones() implemented just for Float32 dtypes");
        }
        else throw std::runtime_error("ones() implemented just for non CPU dtypes");
        auto t = Tensor(storage, shape, compute_contiguous_strides(shape));
        return t;
    }

    // Add random support here.
    Tensor Tensor::randn(Shape shape, DType dtype, Device device) {
        throw std::runtime_error("randn() not implemented");
    }

    // Shape Operations
    Tensor Tensor::view(const Shape& new_shape) const {
        // viewing requires contiguous mem
        if (!is_contiguous()) throw std::runtime_error("view() requires contiguous memory");
        if (shape_numel(new_shape) != numel())
            throw std::runtime_error("View shape has different number of elements");

        // creating the view
        const Strides new_strides = compute_contiguous_strides(new_shape);
        Tensor t(storage_, new_shape, new_strides, storage_offset_);
        t.set_requires_grad(requires_grad());
        return t;
    }

    Tensor Tensor::reshape(const Shape& new_shape) const { throw std::runtime_error("reshape() not implemented"); }

    Tensor Tensor::transpose(const int dim_0, const int dim_1) const {
        // we take out current shape/strides and swap the number at dim0 and dim1
        Shape new_shape = shape_;
        Strides new_strides = strides_;

        std::swap(new_shape[dim_0], new_shape[dim_1]);
        std::swap(new_strides[dim_0], new_strides[dim_1]);

        Tensor t(storage_, new_shape, new_strides, storage_offset_);
        t.set_requires_grad(requires_grad());
        return t;
    }

    Tensor Tensor::contiguous() const { throw std::runtime_error("contiguous() not implemented"); }

    Tensor Tensor::matmul(const Tensor& other) const { throw std::runtime_error("matmul() not implemented"); }

    Tensor Tensor::sum() const { throw std::runtime_error("sum() not implemented"); }

    bool Tensor::is_contiguous() const { return tgrad::is_contiguous(shape_, strides_); }

    void Tensor::print() { throw std::runtime_error("Tensor::print() not implemented"); }
}
