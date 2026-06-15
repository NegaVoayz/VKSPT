#include "render/SwapchainManager.h"
#include <algorithm>
#include <stdexcept>

vk::Image SwapchainManager::ImageAt(uint32_t i) const {
    return m_images[i];
}

SwapchainManager::SwapchainManager(
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physDevice,
    const vk::raii::SurfaceKHR&     surface,
    uint32_t                        computeQf,
    uint32_t                        presentQf,
    uint32_t&                       width,
    uint32_t&                       height)
{
    auto caps = physDevice.getSurfaceCapabilitiesKHR(*surface);
    auto fmts = physDevice.getSurfaceFormatsKHR(*surface);
    auto modes = physDevice.getSurfacePresentModesKHR(*surface);

    auto chosen = pickSurfaceFormat(fmts);
    m_format = chosen.format;

    auto presentMode = pickPresentMode(modes);

    vk::Extent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width  = std::clamp(width,
                                    caps.minImageExtent.width,
                                    caps.maxImageExtent.width);
        extent.height = std::clamp(height,
                                    caps.minImageExtent.height,
                                    caps.maxImageExtent.height);
    }
    width  = extent.width;
    height = extent.height;

    uint32_t imageCount = std::max(caps.minImageCount + 1, 2u);
    if (caps.maxImageCount > 0)
        imageCount = std::min(imageCount, caps.maxImageCount);

    bool concurrent = (computeQf != presentQf);
    std::vector<uint32_t> qfs = {computeQf, presentQf};

    vk::SwapchainCreateInfoKHR sci(
        {}, *surface, imageCount, chosen.format, chosen.colorSpace,
        extent, 1,
        vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eColorAttachment,
        concurrent ? vk::SharingMode::eConcurrent
                   : vk::SharingMode::eExclusive,
        concurrent ? static_cast<uint32_t>(qfs.size()) : 0,
        concurrent ? qfs.data() : nullptr,
        caps.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        presentMode, true, nullptr);
    m_swapchain = vk::raii::SwapchainKHR(device, sci);

    m_images = m_swapchain.getImages();

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_images.size()); ++i) {
        vk::BindImageMemorySwapchainInfoKHR bindSc;
        bindSc.setSwapchain(*m_swapchain);
        bindSc.setImageIndex(i);

        vk::BindImageMemoryInfo bindInfo;
        bindInfo.setImage(m_images[i]);
        bindInfo.setPNext(&bindSc);

        device.bindImageMemory2(bindInfo);
    }

    createImageViews(device, chosen.format);
}

vk::SurfaceFormatKHR SwapchainManager::pickSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& fmts)
{
    vk::SurfaceFormatKHR chosen = fmts[0];
    for (const auto& f : fmts) {
        if (f.format == vk::Format::eR8G8B8A8Srgb &&
            f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            chosen = f; break;
        }
    }
    if (chosen.format != vk::Format::eR8G8B8A8Srgb) {
        for (const auto& f : fmts) {
            if (f.format == vk::Format::eB8G8R8A8Srgb &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                chosen = f; break;
            }
        }
    }
    return chosen;
}

vk::PresentModeKHR SwapchainManager::pickPresentMode(
    const std::vector<vk::PresentModeKHR>& modes)
{
    for (const auto& m : modes)
        if (m == vk::PresentModeKHR::eMailbox)
            return m;
    return vk::PresentModeKHR::eFifo;
}

void SwapchainManager::createImageViews(
    const vk::raii::Device& device, vk::Format format)
{
    for (const auto& img : m_images) {
        vk::ImageViewCreateInfo viewInfo(
            {}, img, vk::ImageViewType::e2D, format,
            vk::ComponentMapping{},
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        m_views.emplace_back(device, viewInfo);
    }
}
