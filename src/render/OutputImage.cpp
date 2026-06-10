#include "render/OutputImage.h"
#include <stdexcept>

void OutputImage::init(const vk::raii::Device&         device,
                       const vk::raii::PhysicalDevice& physDevice,
                       uint32_t width, uint32_t height)
{
    vk::Format format = vk::Format::eR8G8B8A8Unorm;

    vk::ImageCreateInfo imageInfo(
        {}, vk::ImageType::e2D, format,
        {width, height, 1}, 1, 1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive);
    m_image = vk::raii::Image(device, imageInfo);

    // Allocate device-local memory
    auto memReqs = m_image.getMemoryRequirements();
    auto memProps = physDevice.getMemoryProperties();
    uint32_t memTypeIndex = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            memTypeIndex = i; break;
        }
    }
    if (memTypeIndex == 0 &&
        !(memProps.memoryTypes[0].propertyFlags &
          vk::MemoryPropertyFlagBits::eDeviceLocal)) {
        throw std::runtime_error(
            "No device-local memory for output image.");
    }
    vk::MemoryAllocateInfo memInfo(memReqs.size, memTypeIndex);
    m_memory = vk::raii::DeviceMemory(device, memInfo);
    m_image.bindMemory(*m_memory, 0);

    // Image view
    vk::ImageViewCreateInfo viewInfo(
        {}, *m_image, vk::ImageViewType::e2D, format,
        vk::ComponentMapping{},
        vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    m_view = vk::raii::ImageView(device, viewInfo);
}
