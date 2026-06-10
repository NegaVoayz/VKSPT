#pragma once

#include "core/GPUBuffer.h"
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <vector>

/// Builds BLAS from mesh data and TLAS from instance descriptors.
/// Stateless — uses the device's command pool for single-time submits.
class AccelBuilder {
public:
    struct BlasResult {
        vk::raii::AccelerationStructureKHR blas = nullptr;
        vk::DeviceAddress                  addr = 0;
        GPUBuffer                          buf;
    };

    AccelBuilder(const vk::raii::Device& dev,
                 const vk::raii::PhysicalDevice& physDev,
                 uint32_t queueFamily);

    BlasResult buildBLAS(const std::vector<float>&    vertices,
                         const std::vector<uint32_t>& indices);

    void buildTLAS(uint32_t instanceCount,
                   const std::vector<std::array<std::array<float,4>,3>>& transforms,
                   const std::vector<BlasResult>& blasList,
                   GPUBuffer& tlasBuf, GPUBuffer& scratchBuf,
                   GPUBuffer& instBuf,
                   vk::raii::AccelerationStructureKHR& tlas,
                   vk::DeviceAddress& tlasAddr);

private:
    vk::raii::CommandBuffer begin();
    void end(vk::raii::CommandBuffer& cb);

    const vk::raii::Device&         m_dev;
    const vk::raii::PhysicalDevice& m_pdev;
    uint32_t                        m_qf;
    vk::raii::CommandPool           m_pool;
};
