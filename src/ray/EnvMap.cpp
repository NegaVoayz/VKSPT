#include "ray/EnvMap.h"
#include "core/GPUBuffer.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <stdexcept>

namespace {

void createImageResources(const vk::raii::Device&         device,
                          const vk::raii::PhysicalDevice& physDevice,
                          int w, int h,
                          vk::raii::Image&        outImage,
                          vk::raii::DeviceMemory& outMemory,
                          vk::raii::ImageView&    outView,
                          vk::raii::Sampler&      outSampler)
{
    vk::ImageCreateInfo imgInfo({}, vk::ImageType::e2D,
        vk::Format::eR8G8B8A8Unorm, {uint32_t(w), uint32_t(h), 1},
        1, 1, vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eSampled,
        vk::SharingMode::eExclusive);
    outImage = vk::raii::Image(device, imgInfo);

    auto reqs = outImage.getMemoryRequirements();
    auto memProps = physDevice.getMemoryProperties();
    uint32_t memIdx = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((reqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            memIdx = i; break;
        }
    outMemory = vk::raii::DeviceMemory(device,
        vk::MemoryAllocateInfo(reqs.size, memIdx));
    outImage.bindMemory(*outMemory, 0);

    outView = vk::raii::ImageView(device,
        vk::ImageViewCreateInfo({}, *outImage,
            vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm,
            vk::ComponentMapping{},
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
    outSampler = vk::raii::Sampler(device,
        vk::SamplerCreateInfo({},
            vk::Filter::eLinear, vk::Filter::eLinear,
            vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eRepeat,
            vk::SamplerAddressMode::eClampToEdge,
            vk::SamplerAddressMode::eClampToEdge,
            0.0f, false, 1.0f, false, vk::CompareOp::eNever,
            0.0f, 0.0f, vk::BorderColor::eFloatOpaqueBlack));
}

void uploadAndTransition(const vk::raii::Device&   device,
                          const vk::raii::Image&    image,
                          const GPUBuffer&          staging,
                          uint32_t                  queueFamily,
                          int w, int h)
{
    auto cmdPool = vk::raii::CommandPool(device,
        vk::CommandPoolCreateInfo(
            vk::CommandPoolCreateFlagBits::eTransient, queueFamily));
    auto cmdBufs = vk::raii::CommandBuffers(device,
        vk::CommandBufferAllocateInfo(
            *cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cb = cmdBufs[0];
    cb.begin(vk::CommandBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    vk::ImageMemoryBarrier toDst(
        {}, vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        *image, vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, toDst);

    vk::BufferImageCopy region(0, 0, 0,
        vk::ImageSubresourceLayers(
            vk::ImageAspectFlagBits::eColor, 0, 0, 1),
        {0, 0, 0}, {uint32_t(w), uint32_t(h), 1});
    cb.copyBufferToImage(*staging.Buffer, *image,
        vk::ImageLayout::eTransferDstOptimal, region);

    vk::ImageMemoryBarrier toRead(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        *image, vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {}, {}, toRead);
    cb.end();

    auto q = device.getQueue(queueFamily, 0);
    q.submit(vk::SubmitInfo({}, {}, *cb), nullptr);
    q.waitIdle();
}

} // namespace

void EnvMap::load(const vk::raii::Device&         device,
                  const vk::raii::PhysicalDevice& physDevice,
                  uint32_t                        queueFamily,
                  const std::string&              path)
{
    int w, h, ch;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &ch, 4);
    if (!pixels)
        throw std::runtime_error("Failed to load env map: " + path);

    vk::DeviceSize imgSize = static_cast<vk::DeviceSize>(w) * h * 4;
    auto staging = GPUBuffer::Create(
        device, imgSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        physDevice);
    void* mapped = staging.Memory.mapMemory(0, imgSize);
    std::memcpy(mapped, pixels, static_cast<size_t>(imgSize));
    staging.Memory.unmapMemory();
    stbi_image_free(pixels);

    createImageResources(device, physDevice, w, h,
                         m_image, m_memory, m_view, m_sampler);
    uploadAndTransition(device, m_image, staging, queueFamily, w, h);
}
