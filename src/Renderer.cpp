#include "Renderer.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

Renderer::Renderer(
    const vk::raii::Instance&       instance,
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physDevice,
    const vk::raii::SurfaceKHR&     surface,
    uint32_t                        computeQueueFamily,
    uint32_t                        presentQueueFamily,
    const Config&                   config)
    : m_instance(instance)
    , m_device(device)
    , m_physDevice(physDevice)
    , m_config(config)
    , m_computeQueueFamily(computeQueueFamily)
    , m_presentQueueFamily(presentQueueFamily)
{
    createSwapchain(surface);
    createOutputImage();
    createCommandPool();
    createSyncObjects();
}

Renderer::~Renderer() {
    // vk::raii handles cleanup
    m_device.waitIdle();
}

// -----------------------------------------------------------------------------
// Swapchain
// -----------------------------------------------------------------------------
void Renderer::createSwapchain(const vk::raii::SurfaceKHR& surface) {
    auto caps = m_physDevice.getSurfaceCapabilitiesKHR(*surface);
    auto fmts = m_physDevice.getSurfaceFormatsKHR(*surface);
    auto modes = m_physDevice.getSurfacePresentModesKHR(*surface);

    // Pick format: prefer sRGB BGRA8
    vk::SurfaceFormatKHR chosenFmt = fmts[0];
    for (const auto& f : fmts) {
        if (f.format == vk::Format::eB8G8R8A8Srgb &&
            f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            chosenFmt = f;
            break;
        }
    }

    // Pick present mode: FIFO is guaranteed; prefer MAILBOX for low latency
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
    for (const auto& m : modes) {
        if (m == vk::PresentModeKHR::eMailbox) {
            presentMode = m;
            break;
        }
    }

    // Clamp extent
    vk::Extent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width  = std::clamp(m_config.width,
                                   caps.minImageExtent.width,
                                   caps.maxImageExtent.width);
        extent.height = std::clamp(m_config.height,
                                   caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
    }
    m_config.width  = extent.width;
    m_config.height = extent.height;

    uint32_t imageCount = std::max(caps.minImageCount + 1, 2u);
    if (caps.maxImageCount > 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }
    m_swapchainImageCount = imageCount;

    std::vector<uint32_t> queueFamilies = {m_computeQueueFamily, m_presentQueueFamily};
    bool concurrent = (m_computeQueueFamily != m_presentQueueFamily);

    vk::SwapchainCreateInfoKHR swapchainInfo(
        {}, *surface, imageCount, chosenFmt.format, chosenFmt.colorSpace,
        extent, 1,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment,
        concurrent ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        concurrent ? static_cast<uint32_t>(queueFamilies.size()) : 0,
        concurrent ? queueFamilies.data() : nullptr,
        caps.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        presentMode, true, nullptr
    );
    m_swapchain = vk::raii::SwapchainKHR(m_device, swapchainInfo);

    // Get images and create views
    m_swapchainImages = m_swapchain.getImages();
    for (const auto& img : m_swapchainImages) {
        vk::ImageViewCreateInfo viewInfo(
            {}, img, vk::ImageViewType::e2D, chosenFmt.format,
            vk::ComponentMapping{},
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
        );
        m_swapchainViews.emplace_back(m_device, viewInfo);
    }
}

// -----------------------------------------------------------------------------
// Output Storage Image (compute shader target)
// -----------------------------------------------------------------------------
void Renderer::createOutputImage() {
    vk::Format format = vk::Format::eR8G8B8A8Unorm;

    vk::ImageCreateInfo imageInfo(
        {}, vk::ImageType::e2D, format,
        {m_config.width, m_config.height, 1}, 1, 1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive
    );
    m_outputImage = vk::raii::Image(m_device, imageInfo);

    // Allocate device-local memory
    auto memReqs = m_outputImage.getMemoryRequirements();
    auto memTypeIndex = [&]() -> uint32_t {
        auto memProps = m_physDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReqs.memoryTypeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags &
                 vk::MemoryPropertyFlagBits::eDeviceLocal))
            {
                return i;
            }
        }
        throw std::runtime_error("No device-local memory for output image.");
    }();
    vk::MemoryAllocateInfo memInfo(memReqs.size, memTypeIndex);
    m_outputMemory = vk::raii::DeviceMemory(m_device, memInfo);
    m_outputImage.bindMemory(*m_outputMemory, 0);

    // Image view
    vk::ImageViewCreateInfo viewInfo(
        {}, *m_outputImage, vk::ImageViewType::e2D, format,
        vk::ComponentMapping{},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );
    m_outputView = vk::raii::ImageView(m_device, viewInfo);
}

// -----------------------------------------------------------------------------
// Command Pool & Buffer
// -----------------------------------------------------------------------------
void Renderer::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        m_computeQueueFamily
    );
    m_commandPool = vk::raii::CommandPool(m_device, poolInfo);

    vk::CommandBufferAllocateInfo allocInfo(
        *m_commandPool, vk::CommandBufferLevel::ePrimary, 1
    );
    auto bufs  = vk::raii::CommandBuffers(m_device, allocInfo);
    m_computeCmdBuf = std::move(bufs[0]);
}

// -----------------------------------------------------------------------------
// Synchronization
// -----------------------------------------------------------------------------
void Renderer::createSyncObjects() {
    m_imageAvailable = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo{});
    m_renderFinished = vk::raii::Semaphore(m_device, vk::SemaphoreCreateInfo{});
    m_inFlightFence = vk::raii::Fence(m_device,
        vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
}

// -----------------------------------------------------------------------------
// Image layout transition helper
// -----------------------------------------------------------------------------
void Renderer::transitionImageLayout(
    vk::CommandBuffer cmd,
    vk::Image         image,
    vk::ImageLayout   oldLayout,
    vk::ImageLayout   newLayout)
{
    vk::PipelineStageFlags srcStage, dstStage;
    vk::AccessFlags        srcAccess, dstAccess;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eGeneral) {
        srcStage  = vk::PipelineStageFlagBits::eTopOfPipe;
        srcAccess = {};
        dstStage  = vk::PipelineStageFlagBits::eComputeShader;
        dstAccess = vk::AccessFlagBits::eShaderWrite;
    }
    else if (oldLayout == vk::ImageLayout::eGeneral &&
             newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        srcStage  = vk::PipelineStageFlagBits::eComputeShader;
        srcAccess = vk::AccessFlagBits::eShaderWrite;
        dstStage  = vk::PipelineStageFlagBits::eTransfer;
        dstAccess = vk::AccessFlagBits::eTransferRead;
    }
    else if (oldLayout == vk::ImageLayout::eUndefined &&
             newLayout == vk::ImageLayout::eTransferDstOptimal) {
        srcStage  = vk::PipelineStageFlagBits::eTopOfPipe;
        srcAccess = {};
        dstStage  = vk::PipelineStageFlagBits::eTransfer;
        dstAccess = vk::AccessFlagBits::eTransferWrite;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::ePresentSrcKHR) {
        srcStage  = vk::PipelineStageFlagBits::eTransfer;
        srcAccess = vk::AccessFlagBits::eTransferWrite;
        dstStage  = vk::PipelineStageFlagBits::eBottomOfPipe;
        dstAccess = {};
    }
    else {
        throw std::runtime_error("Unsupported layout transition.");
    }

    vk::ImageMemoryBarrier barrier(
        srcAccess, dstAccess,
        oldLayout, newLayout,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        image,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );
    cmd.pipelineBarrier(srcStage, dstStage, {}, {}, {}, barrier);
}

// -----------------------------------------------------------------------------
// Frame Render
// -----------------------------------------------------------------------------
void Renderer::renderFrame(const AccelerationStructure& as, RayTracingPipeline& pipeline) {
    // Wait for previous frame
    if (m_device.waitForFences(*m_inFlightFence, true, UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("Fence wait failed.");
    }
    m_device.resetFences(*m_inFlightFence);

    // Acquire swapchain image
    auto [result, imageIndex] = m_swapchain.acquireNextImage(
        UINT64_MAX, *m_imageAvailable
    );
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("Failed to acquire swapchain image.");
    }

    // Bind descriptor (TLAS + output image)
    pipeline.bindTLAS(as.getTLAS());
    // Use a null sampler since storage images don't use one
    pipeline.bindOutputImage(*m_outputView, nullptr);

    // Record command buffer
    m_computeCmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // Transition output image to GENERAL for compute write
    transitionImageLayout(
        *m_computeCmdBuf, *m_outputImage,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral
    );

    // Bind pipeline + descriptor set
    m_computeCmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    m_computeCmdBuf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        pipeline.getPipelineLayout(), 0,
        pipeline.getDescriptorSet(), nullptr
    );

    // Set push constants (camera)
    struct PushConstants {
        float camOrigin[3]; float _pad0;
        float camU[3];      float _pad1;
        float camV[3];      float _pad2;
        float camW[3];      float _pad3;
    } pc{};
    pc.camOrigin[0] = 0.0f;  pc.camOrigin[1] = 0.0f;  pc.camOrigin[2] = -5.0f;
    // Looking down -Z, simple perspective camera
    float aspect = float(m_config.width) / float(m_config.height);
    float fovTan = 1.0f;  // tan(45 deg) = 1.0 (fov ~90 deg horizontal)
    pc.camU[0] = fovTan * aspect;  pc.camU[1] = 0.0f;  pc.camU[2] = 0.0f;
    pc.camV[0] = 0.0f;             pc.camV[1] = fovTan; pc.camV[2] = 0.0f;
    pc.camW[0] = 0.0f;             pc.camW[1] = 0.0f;    pc.camW[2] = 1.0f;

    m_computeCmdBuf.pushConstants<PushConstants>(
        pipeline.getPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0, pc
    );

    // Dispatch compute
    uint32_t groupsX = (m_config.width  + 7) / 8;
    uint32_t groupsY = (m_config.height + 7) / 8;
    m_computeCmdBuf.dispatch(groupsX, groupsY, 1);

    // Barrier: compute write → transfer read
    transitionImageLayout(
        *m_computeCmdBuf, *m_outputImage,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal
    );

    // Transition swapchain image for copy
    transitionImageLayout(
        *m_computeCmdBuf, m_swapchainImages[imageIndex],
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal
    );

    // Copy output → swapchain
    vk::ImageCopy copyRegion(
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
        {0, 0, 0},
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
        {0, 0, 0},
        {m_config.width, m_config.height, 1}
    );
    m_computeCmdBuf.copyImage(
        *m_outputImage, vk::ImageLayout::eTransferSrcOptimal,
        m_swapchainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
        copyRegion
    );

    // Transition swapchain image for present
    transitionImageLayout(
        *m_computeCmdBuf, m_swapchainImages[imageIndex],
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR
    );

    m_computeCmdBuf.end();

    // Submit
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
    vk::SubmitInfo submitInfo(*m_imageAvailable, waitStage, *m_computeCmdBuf, *m_renderFinished);
    auto computeQueue = m_device.getQueue(m_computeQueueFamily, 0);
    computeQueue.submit(submitInfo, *m_inFlightFence);

    // Present
    vk::PresentInfoKHR presentInfo(*m_renderFinished, *m_swapchain, imageIndex);
    auto presentQueue = m_device.getQueue(m_presentQueueFamily, 0);
    [[maybe_unused]] auto presentResult = presentQueue.presentKHR(presentInfo);
}

// -----------------------------------------------------------------------------
// Save PPM output (read back the compute output image)
// -----------------------------------------------------------------------------
void Renderer::saveOutputPPM(const std::string& path) {
    // Create a host-visible staging buffer big enough for the image
    vk::DeviceSize imgSize = m_config.width * m_config.height * 4; // RGBA8 = 4 bytes

    auto stagingBuf = GPUBuffer::create(
        m_device, imgSize,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );

    // Copy image to buffer
    auto cmdBufAlloc = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*m_commandPool, vk::CommandBufferLevel::ePrimary, 1));
    auto cmdBuf = std::move(cmdBufAlloc[0]);

    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // Transition image to transfer source
    transitionImageLayout(
        *cmdBuf, *m_outputImage,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal
    );

    vk::BufferImageCopy region(
        0, 0, 0,
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
        {0, 0, 0},
        {m_config.width, m_config.height, 1}
    );
    cmdBuf.copyImageToBuffer(
        *m_outputImage, vk::ImageLayout::eTransferSrcOptimal,
        *stagingBuf.buffer, region
    );

    cmdBuf.end();

    auto computeQueue = m_device.getQueue(m_computeQueueFamily, 0);
    vk::SubmitInfo submitInfo({}, {}, *cmdBuf);
    computeQueue.submit(submitInfo, nullptr);
    computeQueue.waitIdle();

    // Read back
    void* mapped = stagingBuf.memory.mapMemory(0, imgSize);
    const uint8_t* pixels = static_cast<const uint8_t*>(mapped);

    // Write PPM (P6 format — binary RGB)
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open output file: " << path << std::endl;
        stagingBuf.memory.unmapMemory();
        return;
    }
    file << "P6\n" << m_config.width << " " << m_config.height << "\n255\n";
    for (uint32_t y = 0; y < m_config.height; ++y) {
        for (uint32_t x = 0; x < m_config.width; ++x) {
            size_t idx = (static_cast<size_t>(y) * m_config.width + x) * 4;
            // RGBA → RGB (flip Y: PPM expects top-left origin, Vulkan has top-left too)
            file.write(reinterpret_cast<const char*>(&pixels[idx]), 3);
        }
    }
    file.close();
    stagingBuf.memory.unmapMemory();

    std::cout << "Saved output to: " << path << std::endl;
}
