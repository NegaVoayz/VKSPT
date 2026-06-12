#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <vector>

/// Owns the swapchain, its images, and image views.
/// Also stores the chosen format for later use (e.g., view creation).
class SwapchainManager {
public:
    SwapchainManager(const vk::raii::Device&         device,
                      const vk::raii::PhysicalDevice& physDevice,
                      const vk::raii::SurfaceKHR&     surface,
                      uint32_t computeQf, uint32_t presentQf,
                      uint32_t& width, uint32_t& height);

    vk::Image                     ImageAt(uint32_t i) const;
    const vk::raii::ImageView&    ViewAt(uint32_t i) const { return m_views[i]; }
    const std::vector<vk::Image>& Images()       const { return m_images; }
    uint32_t                      ImageCount()   const { return static_cast<uint32_t>(m_images.size()); }
    vk::Format          Format()         const { return m_format; }
    const vk::raii::SwapchainKHR& Handle() const { return m_swapchain; }

private:
    static vk::SurfaceFormatKHR pickSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& fmts);
    static vk::PresentModeKHR pickPresentMode(
        const std::vector<vk::PresentModeKHR>& modes);
    void createImageViews(const vk::raii::Device& device, vk::Format format);

    vk::raii::SwapchainKHR          m_swapchain = nullptr;
    std::vector<vk::Image>          m_images;
    std::vector<vk::raii::ImageView> m_views;
    vk::Format                      m_format = vk::Format::eUndefined;
};
