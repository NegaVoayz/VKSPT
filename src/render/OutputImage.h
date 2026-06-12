#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>

/// Compute shader target: a storage image in R8G8B8A8_UNORM format.
class OutputImage {
public:
    OutputImage(const vk::raii::Device&         device,
                 const vk::raii::PhysicalDevice& physDevice,
                 uint32_t width, uint32_t height);

    vk::Image        Handle() const { return *m_image; }
    vk::ImageView    View()   const { return *m_view; }

private:
    vk::raii::Image        m_image  = nullptr;
    vk::raii::DeviceMemory m_memory = nullptr;
    vk::raii::ImageView    m_view   = nullptr;
};
