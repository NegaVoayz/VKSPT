#include "render/FrameCapture.h"
#include "core/GPUBuffer.h"
#include "core/Log.h"
#include <stb_image_write.h>

#include <vector>

FrameCapture::FrameCapture(
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physDevice,
    uint32_t                        queueFamily)
    : m_device(device)
    , m_physDevice(physDevice)
    , m_queueFamily(queueFamily)
{}

void FrameCapture::savePNG(
    const std::string& path,
    vk::Image outputImage,
    uint32_t width, uint32_t height,
    const std::vector<vk::raii::Fence>& inFlightFences,
    const vk::raii::CommandPool& cmdPool)
{
    // Wait for all in-flight GPU work
    std::vector<vk::Fence> fences;
    for (const auto& f : inFlightFences) fences.push_back(*f);
    m_device.waitForFences(fences, true, UINT64_MAX);
    m_device.waitIdle();

    vk::DeviceSize imgSize = width * height * 4;  // RGBA8
    auto staging = GPUBuffer::Create(
        m_device, imgSize,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice);

    auto cbs = vk::raii::CommandBuffers(m_device,
        {*cmdPool, vk::CommandBufferLevel::ePrimary, 1});
    auto& cb = cbs[0];
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ImageMemoryBarrier preBarrier(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eTransferRead,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        outputImage,
        vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, preBarrier);

    cb.copyImageToBuffer(
        outputImage, vk::ImageLayout::eGeneral,
        *staging.Buffer,
        vk::BufferImageCopy(0, 0, 0,
            vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor, 0, 0, 1),
            {0, 0, 0}, {width, height, 1}));

    cb.end();

    auto q = m_device.getQueue(m_queueFamily, 0);
    q.submit(vk::SubmitInfo({}, {}, *cb), nullptr);
    q.waitIdle();

    void* mapped = staging.Memory.mapMemory(0, imgSize);
    int stride = static_cast<int>(width) * 4;
    int r = stbi_write_png(path.c_str(),
        static_cast<int>(width), static_cast<int>(height),
        4, mapped, stride);
    staging.Memory.unmapMemory();

    if (r == 0)
        Log::error("Failed to write output image: {}", path);
    else
        Log::info("Saved output to: {}", path);
}
