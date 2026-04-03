//
// Created by anvesh on 4/3/26.
//

#include "storage.h"
#include <cstdlib>
#include <cstring>

namespace tgrad {
    std::shared_ptr<Storage>
    Storage::allocate(const size_t num_bytes, const Device device, const DType dtype) {
        /*
         * Storage constructor is private std::make_shared is legally not allowed
         * we create a completely empty struct that inherits from Storage exactly at this moment
         * just to bypass the restrictions
         */
        struct MakeSharedEnabler : Storage {};
        const auto storage = std::make_shared<MakeSharedEnabler>();

        storage->dtype_ = dtype;
        storage->device_ = device;
        storage->num_bytes_ = num_bytes;

        if (device == Device::cpu()) {
            storage->data_ = std::malloc(num_bytes);
            if (!storage->data_) throw std::runtime_error("CPU allocation failed. Out of RAM prob.");
            storage->deleter_ = [](void* ptr) { std::free(ptr); };
        }
        else { throw std::runtime_error("CUDA/METAL/OpenCL not implemented yet."); }
        return storage;
    }

    std::shared_ptr<Storage>
    Storage::from_data(const void* data, const size_t num_bytes, const DType dtype) {
        // create mem and fill it with exisiting data
        std::shared_ptr<Storage> storage = allocate(num_bytes, Device::cpu(), dtype);
        std::memcmp(storage->data_ptr(), data, num_bytes);
        return storage;
    }

    Storage::~Storage() { if (data_ && deleter_) deleter_(data_); }

    std::shared_ptr<Storage> Storage::to(const Device target_device) const {
        std::shared_ptr<Storage> new_storage = allocate(num_bytes_, target_device, dtype_);
        if (device_.is_cpu() && target_device.is_cpu())
            std::memcpy(new_storage->data_ptr(), data_, num_bytes_);
        else throw std::runtime_error("CUDA/METAL/OpenCL not implemented yet.");
        return new_storage;
    }
}
