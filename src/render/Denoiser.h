#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>

/// G-buffer storage images + denoise dispatch.
/// Owns normal (rgba16f) and depth (r32f) images.
class Denoiser {
public:
    void init(const vk::raii::Device&         device,
              const vk::raii::PhysicalDevice& physDevice,
              uint32_t width, uint32_t height);

    vk::Image     normalImage() const { return *m_normalImage; }
    vk::Image     depthImage()  const { return *m_depthImage; }
    vk::ImageView normalView()  const { return *m_normalView; }
    vk::ImageView depthView()   const { return *m_depthView; }

private:
    vk::raii::Image        m_normalImage  = nullptr;
    vk::raii::DeviceMemory m_normalMemory = nullptr;
    vk::raii::ImageView    m_normalView   = nullptr;
    vk::raii::Image        m_depthImage   = nullptr;
    vk::raii::DeviceMemory m_depthMemory  = nullptr;
    vk::raii::ImageView    m_depthView    = nullptr;
};
