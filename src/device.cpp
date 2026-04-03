//
// Created by anvesh on 4/2/26.
//
#include "device.h"
#include <stdexcept>

namespace tgrad {
    // returns nice string of device
    std::string Device::str() const {
        switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda:" + std::to_string(index);
        case DeviceType::METAL: return "metal:" + std::to_string(index);
        case DeviceType::OPENCL: return "opencl:" + std::to_string(index);
        default: throw std::runtime_error("Unknown DeviceType!");
        }
    }
}
