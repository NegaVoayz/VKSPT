#pragma once

#include <vulkan/vulkan_raii.hpp>

/// RAII pair: vk::Buffer + vk::DeviceMemory.
/// Helper struct — not a standalone functional class.
struct GPUBuffer {
    vk::raii::Buffer       buffer     = nullptr;
    vk::raii::DeviceMemory memory     = nullptr;
    vk::DeviceSize          size      = 0;
    vk::DeviceAddress       address   = 0;   // requires bufferDeviceAddress

    /// Allocate a GPU buffer with the given usage and memory properties.
    /// Returns a fully initialized GPUBuffer.
    static GPUBuffer create(
        const vk::raii::Device&         device,
        vk::DeviceSize                  size,
        vk::BufferUsageFlags            usage,
        vk::MemoryPropertyFlags         memProps,
        const vk::raii::PhysicalDevice& physDevice
    );

    /// Create a buffer + memory pair for a staging buffer (host-visible, coherent).
    /// Copies data immediately via map/unmap.
    static GPUBuffer createStaging(
        const vk::raii::Device&         device,
        const void*                     data,
        vk::DeviceSize                  size,
        vk::BufferUsageFlags            usage,
        const vk::raii::PhysicalDevice& physDevice
    );
};
