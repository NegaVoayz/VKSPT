#include "render/Denoiser.h"

void Denoiser::init(const vk::raii::Device&         device,
                    const vk::raii::PhysicalDevice& physDevice,
                    uint32_t width, uint32_t height)
{
    auto createImage = [&](vk::Format format, vk::raii::Image& img,
                            vk::raii::DeviceMemory& mem,
                            vk::raii::ImageView& view) {
        vk::ImageCreateInfo info({}, vk::ImageType::e2D, format,
            {width, height, 1}, 1, 1, vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eStorage,
            vk::SharingMode::eExclusive);
        img = vk::raii::Image(device, info);
        auto reqs = img.getMemoryRequirements();
        auto props = physDevice.getMemoryProperties();
        uint32_t idx = 0;
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
            if ((reqs.memoryTypeBits & (1u << i)) &&
                (props.memoryTypes[i].propertyFlags &
                 vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                idx = i; break;
            }
        mem = vk::raii::DeviceMemory(
            device, vk::MemoryAllocateInfo(reqs.size, idx));
        img.bindMemory(*mem, 0);
        view = vk::raii::ImageView(device,
            vk::ImageViewCreateInfo({}, *img,
                vk::ImageViewType::e2D, format,
                vk::ComponentMapping{},
                vk::ImageSubresourceRange(
                    vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
    };
    createImage(vk::Format::eR16G16B16A16Sfloat,
                m_normalImage, m_normalMemory, m_normalView);
    createImage(vk::Format::eR32Sfloat,
                m_depthImage, m_depthMemory, m_depthView);
}
