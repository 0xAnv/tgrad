//
// Created by anvesh on 4/3/26.
//

#pragma once

#include "device.h"
#include "dtype.h"
#include <functional>
#include <memory>

#ifndef TGRAD_STORAGE_H
#define TGRAD_STORAGE_H

namespace tgrad {
    // storage owns a raw blob of bytes on a specific device
    // Multiple tensors can share same storage (devices) via shared ptr
    class Storage {
    public:
        // new empty space
        static std::shared_ptr<Storage> allocate(size_t num_bytes, Device device, DType dtype);
        static std::shared_ptr<Storage> from_data(const void* data, size_t num_bytes, DType dtype);
        std::shared_ptr<Storage> to(Device target_device) const;

        // destructor
        ~Storage();

        // raw ptr access for cuda/opencl/metal & getters
        const void* data_ptr() const { return data_; }
        void* data_ptr() { return data_; }
        size_t num_bytes() const { return num_bytes_; }
        Device device() const { return device_; }
        DType dtype() const { return dtype_; }

        // type safe access
        template <typename T>
        T* data_as() { return static_cast<T*>(data_); }

    private:
        // private alloc,we must use it statically
        Storage() = default;
        Device device_ = Device::cpu();
        DType dtype_ = DType::Float32;
        void* data_ = nullptr; // actual raw bytes in memory
        size_t num_bytes_ = 0;

        // custom deleter to free this memory
        std::function<void(void*)> deleter_; // takes void* returns void
    };
}


#endif //TGRAD_STORAGE_H
