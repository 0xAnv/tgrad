//
// Created by anvesh on 4/2/26.
//

#pragma  once
#include <cstdint>
#include <string>

#ifndef TGRAD_DEVICE_H
#define TGRAD_DEVICE_H

namespace tgrad
{
    // types of hardware
    enum class DeviceType : uint8_t {
        CPU, // AVX etc
        CUDA, // nvidia-gpus
        METAL, // mps
        OPENCL // felt to include for legacy
    };

    struct Device {
        DeviceType type;
        int32_t index;

        // constructor defaults to cpu
        Device() : type(DeviceType::CPU), index(0) {}
        explicit Device(const DeviceType t, const int32_t idx = 0) : type(t), index(idx) {}

        // Factory methods for creating tensors eg Tensor::zeros({1,1})
        static Device cpu() { return Device(DeviceType::CPU); }
        static Device cuda() { return Device(DeviceType::CUDA); }

        [[nodiscard]]
        bool is_cuda() const { return type == DeviceType::CUDA; }

        [[nodiscard]]
        bool is_cpu() const { return type == DeviceType::CPU; }

        [[nodiscard]]
        bool is_metal() const { return type == DeviceType::METAL; }

        [[nodiscard]]
        bool is_opencl() const { return type == DeviceType::OPENCL; }

        // operator overloading to check if devices are same
        bool operator==(const Device& other) const {
            return type == other.type && index == other.index;
        }

        [[nodiscard]]
        std::string str() const; // returns string like cpu, cuda:0
    };
}

#endif //TGRAD_DEVICE_H
