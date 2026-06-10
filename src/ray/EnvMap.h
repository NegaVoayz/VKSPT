#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <string>

/// Loads an equirectangular environment map from JPEG/PNG, creating
/// a device-local VkImage + ImageView + Sampler for shader access.
class EnvMap {
public:
    void load(const vk::raii::Device&         device,
              const vk::raii::PhysicalDevice& physDevice,
              uint32_t                        queueFamily,
              const std::string&              path);

    vk::ImageView view()    const { return *m_view; }
    vk::Sampler   sampler() const { return *m_sampler; }

private:
    vk::raii::Image        m_image   = nullptr;
    vk::raii::DeviceMemory m_memory  = nullptr;
    vk::raii::ImageView    m_view    = nullptr;
    vk::raii::Sampler      m_sampler = nullptr;
};
