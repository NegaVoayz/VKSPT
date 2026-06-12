#pragma once

#include <vulkan/vulkan_raii.hpp>

/// RAII pair: vk::Buffer + vk::DeviceMemory.
/// Helper struct — not a standalone functional class.
struct GPUBuffer {
    vk::raii::Buffer       Buffer     = nullptr;
    vk::raii::DeviceMemory Memory     = nullptr;
    vk::DeviceSize          Size      = 0;
    vk::DeviceAddress       Address   = 0;

    /// Allocate a GPU buffer with the given usage and memory properties.
    /// Returns a fully initialized GPUBuffer.
    static GPUBuffer Create(
        const vk::raii::Device&         device,
        vk::DeviceSize                  size,
        vk::BufferUsageFlags            usage,
        vk::MemoryPropertyFlags         memProps,
        const vk::raii::PhysicalDevice& physDevice
    );

    /// Host-visible staging buffer: allocate + map + copy + unmap.
    static GPUBuffer CreateStaging(
        const vk::raii::Device&         device,
        const void*                     data,
        vk::DeviceSize                  size,
        vk::BufferUsageFlags            usage,
        const vk::raii::PhysicalDevice& physDevice
    );
};
