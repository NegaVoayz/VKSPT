#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <string>
#include <vector>

/// Save compute output image to PNG file on disk.
class FrameCapture {
public:
    FrameCapture(const vk::raii::Device&         device,
                 const vk::raii::PhysicalDevice& physDevice,
                 uint32_t                        queueFamily);

    /// Read back the output image via staging buffer and write to PNG.
    void savePNG(const std::string& path,
                 vk::Image outputImage,
                 uint32_t width, uint32_t height,
                 const std::vector<vk::raii::Fence>& inFlightFences,
                 const vk::raii::CommandPool& cmdPool);

private:
    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    uint32_t                        m_queueFamily;
};
