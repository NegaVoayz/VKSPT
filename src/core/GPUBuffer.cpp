#include "core/GPUBuffer.h"

#include <cstring>
#include <stdexcept>

GPUBuffer GPUBuffer::Create(
    const vk::raii::Device&         device,
    vk::DeviceSize                  size,
    vk::BufferUsageFlags            usage,
    vk::MemoryPropertyFlags         memProps,
    const vk::raii::PhysicalDevice& physDevice)
{
    GPUBuffer result;
    result.Size = size;

    vk::BufferCreateInfo bufferInfo({}, size, usage);
    result.Buffer = vk::raii::Buffer(device, bufferInfo);

    auto memRequirements = result.Buffer.getMemoryRequirements();
    auto memTypeIndex    = [&]() -> uint32_t {
        auto memProperties = physDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((memRequirements.memoryTypeBits & (1u << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & memProps) == memProps)
            {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type.");
    }();

    vk::MemoryAllocateFlagsInfo allocFlags(
        vk::MemoryAllocateFlagBits::eDeviceAddress
    );
    vk::MemoryAllocateInfo allocInfo(memRequirements.size, memTypeIndex, &allocFlags);
    result.Memory = vk::raii::DeviceMemory(device, allocInfo);

    result.Buffer.bindMemory(*result.Memory, 0);

    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        vk::BufferDeviceAddressInfo addrInfo(*result.Buffer);
        result.Address = device.getBufferAddress(addrInfo);
    }

    return result;
}

GPUBuffer GPUBuffer::CreateStaging(
    const vk::raii::Device&         device,
    const void*                     data,
    vk::DeviceSize                  size,
    vk::BufferUsageFlags            usage,
    const vk::raii::PhysicalDevice& physDevice)
{
    auto buf = Create(
        device, size,
        usage | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        physDevice
    );

    void* mapped = buf.Memory.mapMemory(0, size);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    buf.Memory.unmapMemory();

    return buf;
}
