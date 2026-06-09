#include "Renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstring>
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
    createCommandBuffers();
    createSyncObjects();

    // Timestamp queries for GPU profiling
    auto queueProps = physDevice.getQueueFamilyProperties();
    auto& qf = queueProps[computeQueueFamily];
    if (qf.timestampValidBits > 0) {
        vk::QueryPoolCreateInfo qpInfo({}, vk::QueryType::eTimestamp, MAX_FRAMES_IN_FLIGHT * TIMESTAMPS_PER_FRAME);
        m_timestampPool   = vk::raii::QueryPool(device, qpInfo);
        m_timestampPeriod = physDevice.getProperties().limits.timestampPeriod; // nanoseconds
        m_hasTimestamps   = true;
    }
}

Renderer::~Renderer() {
    m_device.waitIdle();
    // vk::raii handles cleanup; swapchain destroyed before surface (member order)
}

void Renderer::initSortedPipeline(RayTracingPipeline& pipeline) {
    m_raySorter = std::make_unique<RaySorter>(
        m_device, m_physDevice, m_config.width, m_config.height, 1  // SPP=1
    );
    pipeline.createSortPipeline("shaders/raytrace_sort.comp.spv");
    pipeline.createNormalizePipeline("shaders/normalize.comp.spv");
    pipeline.createClassifyPipeline("shaders/raytrace_classify.comp.spv");
    pipeline.createProcessPipeline("shaders/raytrace_process.comp.spv");
    m_useSorting = true;
    std::cout << "  Sorted ray pipeline initialized." << std::endl;
}

// -----------------------------------------------------------------------------
// Swapchain
// -----------------------------------------------------------------------------
void Renderer::createSwapchain(const vk::raii::SurfaceKHR& surface) {
    auto caps = m_physDevice.getSurfaceCapabilitiesKHR(*surface);
    auto fmts = m_physDevice.getSurfaceFormatsKHR(*surface);
    auto modes = m_physDevice.getSurfacePresentModesKHR(*surface);

    // Pick format: prefer R8G8B8A8_SRGB to avoid R/B swap with compute output.
    // Fall back to B8G8R8A8_SRGB if unavailable.
    vk::SurfaceFormatKHR chosenFmt = fmts[0];
    for (const auto& f : fmts) {
        if (f.format == vk::Format::eR8G8B8A8Srgb &&
            f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            chosenFmt = f;
            break;
        }
    }
    if (chosenFmt.format != vk::Format::eR8G8B8A8Srgb) {
        for (const auto& f : fmts) {
            if (f.format == vk::Format::eB8G8R8A8Srgb &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                chosenFmt = f;
                break;
            }
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

    // Get images
    m_swapchainImages = m_swapchain.getImages();

    // Vulkan 1.4: Bind swapchain image memory explicitly
    for (uint32_t i = 0; i < m_swapchainImageCount; ++i) {
        vk::BindImageMemorySwapchainInfoKHR bindInfo;
        bindInfo.setSwapchain(*m_swapchain);
        bindInfo.setImageIndex(i);

        vk::BindImageMemoryInfo bindMemInfo;
        bindMemInfo.setImage(m_swapchainImages[i]);
        bindMemInfo.setPNext(&bindInfo);

        m_device.bindImageMemory2(bindMemInfo);
    }

    // Create views
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
// Output Storage Image (compute shader target) — stays in GENERAL layout
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
// Command Pool & Buffers (one per frame-in-flight)
// -----------------------------------------------------------------------------
void Renderer::createCommandBuffers() {
    vk::CommandPoolCreateInfo poolInfo(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        m_computeQueueFamily
    );
    m_commandPool = vk::raii::CommandPool(m_device, poolInfo);

    vk::CommandBufferAllocateInfo allocInfo(
        *m_commandPool, vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT
    );
    m_commandBuffers = vk::raii::CommandBuffers(m_device, allocInfo);
}

// -----------------------------------------------------------------------------
// Synchronization (per frame-in-flight)
// -----------------------------------------------------------------------------
void Renderer::createSyncObjects() {
    // imageAvailable: per-frame-in-flight
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_imageAvailableSem.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    }
    // renderFinished: per-swapchain-image (must be unique to avoid reuse)
    for (uint32_t i = 0; i < m_swapchainImageCount; ++i) {
        m_renderFinishedSem.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    }
    // inFlightFences: per-frame-in-flight
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_inFlightFences.emplace_back(m_device,
            vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
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
             newLayout == vk::ImageLayout::eGeneral) {
        // No transition needed — already in correct layout.
        // Still issue a barrier for write-after-read/write-after-write.
        srcStage  = vk::PipelineStageFlagBits::eComputeShader;
        srcAccess = vk::AccessFlagBits::eShaderWrite;
        dstStage  = vk::PipelineStageFlagBits::eComputeShader;
        dstAccess = vk::AccessFlagBits::eShaderWrite;
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
    uint32_t frameIdx = m_currentFrame % MAX_FRAMES_IN_FLIGHT;

    // Wait for this frame's fence
    if (m_device.waitForFences(*m_inFlightFences[frameIdx], true, UINT64_MAX) != vk::Result::eSuccess) {
        throw std::runtime_error("Fence wait failed.");
    }
    m_device.resetFences(*m_inFlightFences[frameIdx]);

    // Acquire swapchain image (frame-indexed acquire semaphore)
    auto [result, imageIndex] = m_swapchain.acquireNextImage(
        UINT64_MAX, *m_imageAvailableSem[frameIdx]
    );
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("Failed to acquire swapchain image.");
    }

    // ===== Phase 4.5: Sorted ray tracing pipeline (first frame only) =====
    if (m_useSorting && m_currentFrame == 0) {
        float aspect = float(m_config.width) / float(m_config.height);
        float fovTan = 1.0f;
        float camOrigin[3] = {0, 0, -5};
        float camU[3] = {fovTan * aspect, 0, 0};
        float camV[3] = {0, fovTan, 0};
        float camW[3] = {0, 0, 1};

        // 1. Init rays
        m_raySorter->initRays(camOrigin, camU, camV, camW, fovTan);

        // 2. Zero pixel accum buffer
        {
            uint32_t pixelCount = m_config.width * m_config.height;
            std::vector<RaySorter::PixelEntry> zeros(pixelCount, {});
            auto zeroStaging = GPUBuffer::createStaging(m_device, zeros.data(),
                pixelCount * sizeof(RaySorter::PixelEntry),
                vk::BufferUsageFlagBits::eTransferSrc, m_physDevice);
            vk::raii::CommandPool pool(m_device,
                {vk::CommandPoolCreateFlagBits::eTransient, m_computeQueueFamily});
            auto cbs = vk::raii::CommandBuffers(m_device,
                {*pool, vk::CommandBufferLevel::ePrimary, 1});
            cbs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
            vk::BufferCopy rgn(0, 0, pixelCount * sizeof(RaySorter::PixelEntry));
            cbs[0].copyBuffer(*zeroStaging.buffer,
                *m_raySorter->getAccumBuffer().buffer, rgn);
            cbs[0].end();
            auto q = m_device.getQueue(m_computeQueueFamily, 0);
            vk::SubmitInfo subInfo; vk::CommandBuffer cb = *cbs[0]; subInfo.commandBufferCount=1; subInfo.pCommandBuffers=&cb; q.submit(subInfo, nullptr);
            q.waitIdle();
        }

        // 3. Transition output image for compute writes
        {
            vk::raii::CommandPool pool(m_device,
                {vk::CommandPoolCreateFlagBits::eTransient, m_computeQueueFamily});
            auto cbs = vk::raii::CommandBuffers(m_device,
                {*pool, vk::CommandBufferLevel::ePrimary, 1});
            cbs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
            transitionImageLayout(*cbs[0], *m_outputImage,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
            cbs[0].end();
            auto q = m_device.getQueue(m_computeQueueFamily, 0);
            vk::SubmitInfo subInfo; vk::CommandBuffer cb = *cbs[0]; subInfo.commandBufferCount=1; subInfo.pCommandBuffers=&cb; q.submit(subInfo, nullptr);
            q.waitIdle();
        }

        // 4. Bind ALL descriptors (using frame 0 descriptor set)
        pipeline.bindTLAS(0, as.getTLAS());
        pipeline.bindOutputImage(0, *m_outputView, nullptr);
        pipeline.bindMaterialBuffer(0, *as.getMaterialBuffer().buffer,
                                     as.getMaterialBuffer().size);
        pipeline.bindLightBuffer(0, *as.getLightBuffer().buffer,
                                  as.getLightBuffer().size);
        pipeline.bindGeometrySSBOs(0,
            *as.getVertexDataBuffer().buffer, as.getVertexDataBuffer().size,
            *as.getIndexDataBuffer().buffer,  as.getIndexDataBuffer().size,
            *as.getRangeBuffer().buffer,      as.getRangeBuffer().size);
        pipeline.bindRayBuffer(0, *m_raySorter->getRayBuffer().buffer,
                                m_raySorter->getRayBuffer().size);
        pipeline.bindCounterBuffer(0, *m_raySorter->getCounterBuffer().buffer,
                                    m_raySorter->getCounterBuffer().size);
        pipeline.bindPixelAccum(0, *m_raySorter->getAccumBuffer().buffer,
                                 m_raySorter->getAccumBuffer().size);
        pipeline.bindOverflowBuffer(0, *m_raySorter->getOverflowBuffer().buffer,
                                     m_raySorter->getOverflowBuffer().size);

        // Zero overflow counter before starting
        m_raySorter->drainOverflow();

        // 5. Push constants
        struct SortPC {
            float camOrigin[3]; float _p0;
            float camU[3]; float _p1;
            float camV[3]; float _p2;
            float camW[3]; float _p3;
            int spp, maxBounces, matCount;
            float ft, sm, fsw;
            int scat; float mt;
        } pc{};
        pc.camOrigin[0]=0; pc.camOrigin[1]=0; pc.camOrigin[2]=-5;
        pc.camU[0]=fovTan*aspect; pc.camV[1]=fovTan; pc.camW[2]=1;
        pc.spp=1; pc.maxBounces=24;
        pc.matCount=(int)as.getMaterialCount();
        pc.ft=fovTan; pc.sm=1.0f; pc.fsw=40.0f;
        pc.scat=2; pc.mt=0.9999f;

        // 6. Chunked two-stage dispatch with operation sorting.
        //    Splits large generations into 128K chunks to avoid TDR timeout.
        //    Each chunk: classify (trace+cache) → CPU sort by RayAction → process.
        //    Sorting consecutive rays by action reduces warp divergence.
        auto queue = m_device.getQueue(m_computeQueueFamily, 0);
        uint32_t head = 0;
        int iters = 0;
        const uint32_t LOCAL = 64;
        const uint32_t CHUNK = 128u * 1024u;  // max rays per chunk

        auto dispatchPipe = [&](vk::Pipeline pipe, uint32_t groups) {
            vk::raii::CommandPool pool(m_device,
                {vk::CommandPoolCreateFlagBits::eTransient |
                    vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                 m_computeQueueFamily});
            auto cbs = vk::raii::CommandBuffers(m_device,
                {*pool, vk::CommandBufferLevel::ePrimary, 1});
            cbs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
            cbs[0].bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
            cbs[0].bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                pipeline.getPipelineLayout(), 0, pipeline.getDescriptorSet(0), nullptr);
            cbs[0].pushConstants<SortPC>(pipeline.getPipelineLayout(),
                vk::ShaderStageFlagBits::eCompute, 0, pc);
            cbs[0].dispatch(groups, 1, 1);
            cbs[0].end();
            queue.submit({vk::SubmitInfo().setCommandBuffers(*cbs[0])}, nullptr);
            queue.waitIdle();
        };

        std::cout << "[Sort] Starting chunked two-stage loop (chunk=" << CHUNK << ")..." << std::endl;
        for (int iter = 0; iter < 512; ++iter) {
            uint32_t tail = m_raySorter->getTailCount();
            if (tail > RaySorter::MAX_RAYS) tail = RaySorter::MAX_RAYS;
            uint32_t active = (tail > head) ? (tail - head) : 0;
            if (active == 0) {
                if (m_raySorter->stashedCount() == 0) break;
                head = 0; m_raySorter->resetCounters();
                m_raySorter->injectStashed(std::min(RaySorter::MAX_RAYS, CHUNK));
                tail = m_raySorter->getTailCount();
                active = (tail > head) ? (tail - head) : 0;
                if (active == 0) continue;
            }

            uint32_t N = (active < CHUNK) ? active : CHUNK;
            uint32_t processEnd = head + N;
            m_raySorter->advanceHead(head);
            uint32_t groups = (N + LOCAL - 1) / LOCAL;

            if (iter % 20 == 0 || N < active) {
                std::cout << "[Sort] iter " << iter << " head=" << head
                          << " tail=" << tail << " chunk=" << N
                          << " remain=" << (active - N) << std::endl;
            }

            // Stage A: Classify (trace, cache hit info, set RayAction)
            dispatchPipe(pipeline.getClassifyPipeline(), groups);

            // Stage B: Sort by RayAction (CPU-side, <1ms for 128K rays)
            m_raySorter->sortBatchByAction(head, N);

            // Stage C: Process (read cached hit info, no re-trace)
            dispatchPipe(pipeline.getProcessPipeline(), groups);

            iters++;

            uint32_t overflowed = m_raySorter->drainOverflow();
            if (overflowed > 0) { m_raySorter->clampTail(); }

            uint32_t newTail = m_raySorter->getTailCount();
            head = processEnd;
            if (head >= newTail && m_raySorter->stashedCount() == 0) break;
        }
        std::cout << "[Sort] " << iters << " iterations, head=" << head
                  << " tail=" << m_raySorter->getTailCount()
                  << " stashed=" << m_raySorter->stashedCount() << std::endl;

        // 7. Normalize pass
        {
            vk::raii::CommandPool pool(m_device,
                {vk::CommandPoolCreateFlagBits::eTransient, m_computeQueueFamily});
            auto cbs = vk::raii::CommandBuffers(m_device,
                {*pool, vk::CommandBufferLevel::ePrimary, 1});
            cbs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
            cbs[0].bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getNormalizePipeline());
            cbs[0].bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                pipeline.getPipelineLayout(), 0, pipeline.getDescriptorSet(0), nullptr);
            uint32_t gx = (m_config.width + 7) / 8;
            uint32_t gy = (m_config.height + 7) / 8;
            cbs[0].dispatch(gx, gy, 1);
            cbs[0].end();
            queue.submit({vk::SubmitInfo().setCommandBuffers(*cbs[0])}, nullptr);
            queue.waitIdle();
        }

        // 8. Copy output to swapchain + present
        {
            auto& cb = m_commandBuffers[frameIdx];
            cb.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

            transitionImageLayout(*cb, m_swapchainImages[imageIndex],
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

            vk::ImageCopy copyRgn(
                vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                {0,0,0},
                vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                {0,0,0}, {m_config.width, m_config.height, 1}
            );
            cb.copyImage(*m_outputImage, vk::ImageLayout::eGeneral,
                m_swapchainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal, copyRgn);

            transitionImageLayout(*cb, m_swapchainImages[imageIndex],
                vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);
            cb.end();

            vk::PipelineStageFlags ws = vk::PipelineStageFlagBits::eTransfer;
            vk::SubmitInfo swapSI;
            vk::Semaphore waitSem = *m_imageAvailableSem[frameIdx];
            vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
            swapSI.waitSemaphoreCount = 1;
            swapSI.pWaitSemaphores = &waitSem;
            swapSI.pWaitDstStageMask = &waitStage;
            vk::CommandBuffer swapCB = *cb;
            swapSI.commandBufferCount = 1;
            swapSI.pCommandBuffers = &swapCB;
            vk::Semaphore sigSem = *m_renderFinishedSem[imageIndex];
            swapSI.signalSemaphoreCount = 1;
            swapSI.pSignalSemaphores = &sigSem;
            queue.submit(swapSI, *m_inFlightFences[frameIdx]);

            vk::PresentInfoKHR pi(*m_renderFinishedSem[imageIndex], *m_swapchain, imageIndex);
            m_device.getQueue(m_presentQueueFamily, 0).presentKHR(pi);
        }

        m_device.waitForFences(*m_inFlightFences[frameIdx], true, UINT64_MAX);
        m_currentFrame++;
        std::cout << "[Sort] Frame complete." << std::endl;
        return;
    }

    // Bind descriptor (TLAS + output image + material buffer + light buffer) for this frame index
    pipeline.bindTLAS(frameIdx, as.getTLAS());
    pipeline.bindOutputImage(frameIdx, *m_outputView, nullptr);
    pipeline.bindMaterialBuffer(frameIdx, *as.getMaterialBuffer().buffer,
                                 as.getMaterialBuffer().size);
    pipeline.bindLightBuffer(frameIdx, *as.getLightBuffer().buffer,
                              as.getLightBuffer().size);
    pipeline.bindGeometrySSBOs(frameIdx,
        *as.getVertexDataBuffer().buffer, as.getVertexDataBuffer().size,
        *as.getIndexDataBuffer().buffer,  as.getIndexDataBuffer().size,
        *as.getRangeBuffer().buffer,      as.getRangeBuffer().size);

    // Record command buffer
    auto& cmdBuf = m_commandBuffers[frameIdx];
    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // Transition output image: first frame Undefined→General, subsequent frames barrier only
    if (m_currentFrame == 0) {
        transitionImageLayout(
            *cmdBuf, *m_outputImage,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral
        );
    } else {
        transitionImageLayout(
            *cmdBuf, *m_outputImage,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral
        );
    }

    // Bind pipeline + descriptor set (per-frame descriptor set)
    cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.getPipeline());
    cmdBuf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        pipeline.getPipelineLayout(), 0,
        pipeline.getDescriptorSet(frameIdx), nullptr
    );

    // Set push constants (camera + spectral params) — 96 bytes
    struct PushConstants {
        float camOrigin[3]; float _pad0;
        float camU[3];      float _pad1;
        float camV[3];      float _pad2;
        float camW[3];      float _pad3;
        int   samplesPerPixel;
        int   maxBounces;
        int   materialCount;
        float fovTan;
        float splitMult;
        float forceSplitWidth;
        int   scatterSamples;
        float mergeThreshold;
    } pc{};
    pc.camOrigin[0] = 0.0f;  pc.camOrigin[1] = 0.0f;  pc.camOrigin[2] = -5.0f;
    float aspect = float(m_config.width) / float(m_config.height);
    float fovTan = 1.0f;
    pc.camU[0] = fovTan * aspect;  pc.camU[1] = 0.0f;  pc.camU[2] = 0.0f;
    pc.camV[0] = 0.0f;             pc.camV[1] = fovTan; pc.camV[2] = 0.0f;
    pc.camW[0] = 0.0f;             pc.camW[1] = 0.0f;    pc.camW[2] = 1.0f;
    pc.samplesPerPixel = 4;  // SPP=4 production
    pc.maxBounces      = 24;
    pc.materialCount   = static_cast<int>(as.getMaterialCount());
    pc.fovTan          = 1.0f;
    pc.splitMult       = 1.0f;
    pc.forceSplitWidth = 40.0f;
    pc.scatterSamples  = 2;
    pc.mergeThreshold  = 0.9999f;
    pc.maxBounces      = 24;
    pc.materialCount   = static_cast<int>(as.getMaterialCount());
    pc.fovTan          = 1.0f;
    pc.splitMult       = 1.0f;
    pc.forceSplitWidth = 40.0f;
    pc.scatterSamples  = 2;
    pc.mergeThreshold  = 0.9999f;

    cmdBuf.pushConstants<PushConstants>(
        pipeline.getPipelineLayout(),
        vk::ShaderStageFlagBits::eCompute,
        0, pc
    );

    // Timestamp: reset + write start
    if (m_hasTimestamps) {
        cmdBuf.resetQueryPool(*m_timestampPool, frameIdx * TIMESTAMPS_PER_FRAME, TIMESTAMPS_PER_FRAME);
        cmdBuf.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                              *m_timestampPool, frameIdx * TIMESTAMPS_PER_FRAME);
    }

    // Dispatch compute
    uint32_t groupsX = (m_config.width  + 7) / 8;
    uint32_t groupsY = (m_config.height + 7) / 8;
    cmdBuf.dispatch(groupsX, groupsY, 1);

    // Timestamp: write end
    if (m_hasTimestamps) {
        cmdBuf.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                              *m_timestampPool, frameIdx * TIMESTAMPS_PER_FRAME + 1);
    }

    // Barrier: compute write → transfer read (keep GENERAL layout)
    vk::ImageMemoryBarrier outputBarrier(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eTransferRead,
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        *m_outputImage,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );
    cmdBuf.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, outputBarrier
    );

    // Transition swapchain image for copy
    transitionImageLayout(
        *cmdBuf, m_swapchainImages[imageIndex],
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
    cmdBuf.copyImage(
        *m_outputImage, vk::ImageLayout::eGeneral,
        m_swapchainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
        copyRegion
    );

    // Transition swapchain image for present
    transitionImageLayout(
        *cmdBuf, m_swapchainImages[imageIndex],
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR
    );

    cmdBuf.end();

    // Submit: wait on frame-indexed acquire semaphore, signal image-indexed render semaphore
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
    vk::SubmitInfo submitInfo(
        *m_imageAvailableSem[frameIdx], waitStage,
        *cmdBuf, *m_renderFinishedSem[imageIndex]
    );
    auto computeQueue = m_device.getQueue(m_computeQueueFamily, 0);
    computeQueue.submit(submitInfo, *m_inFlightFences[frameIdx]);

    // Present
    vk::PresentInfoKHR presentInfo(*m_renderFinishedSem[imageIndex], *m_swapchain, imageIndex);
    auto presentQueue = m_device.getQueue(m_presentQueueFamily, 0);
    [[maybe_unused]] auto presentResult = presentQueue.presentKHR(presentInfo);

    // For the first frame, block until GPU work is done so PPM save reads correctly.
    // Subsequent frames overlap CPU/GPU via fences waited at start of next frame.
    if (m_currentFrame == 0) {
        m_device.waitForFences(*m_inFlightFences[frameIdx], true, UINT64_MAX);
    }

    // Timestamp readback & reporting every 60 frames
    if (m_hasTimestamps && m_currentFrame >= 2) {
        uint32_t prevFrame = (m_currentFrame - 1) % MAX_FRAMES_IN_FLIGHT;
        uint64_t results[TIMESTAMPS_PER_FRAME];
        vk::Result r = static_cast<vk::Device>(*m_device).getQueryPoolResults(
            *m_timestampPool,
            prevFrame * TIMESTAMPS_PER_FRAME, TIMESTAMPS_PER_FRAME,
            sizeof(results), results, sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait
        );
        if (r == vk::Result::eSuccess && results[0] > 0 && results[1] > results[0]) {
            float gpuTimeMs = float(results[1] - results[0]) * m_timestampPeriod / 1e6f;
            m_frameCount++;
            if (m_frameCount % 60 == 0) {
                std::cout << "[GPU] frame " << m_frameCount
                          << " compute time: " << gpuTimeMs << " ms" << std::endl;
            }
        }
    }

    m_currentFrame++;
}

// -----------------------------------------------------------------------------
// Save PNG output (read back the compute output image)
// -----------------------------------------------------------------------------
void Renderer::saveOutputPNG(const std::string& path) {
    // Wait for all in-flight GPU work to complete
    std::vector<vk::Fence> fences;
    for (auto& f : m_inFlightFences) {
        fences.push_back(*f);
    }
    m_device.waitForFences(fences, true, UINT64_MAX);
    m_device.waitIdle();  // belt and suspenders: also wait for present queue

    // Create a host-visible staging buffer big enough for the image
    vk::DeviceSize imgSize = m_config.width * m_config.height * 4; // RGBA8

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

    // Barrier: ensure compute writes are visible
    vk::ImageMemoryBarrier preBarrier(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eTransferRead,
        vk::ImageLayout::eGeneral,
        vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        *m_outputImage,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );
    cmdBuf.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, preBarrier
    );

    vk::BufferImageCopy region(
        0, 0, 0,
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
        {0, 0, 0},
        {m_config.width, m_config.height, 1}
    );
    cmdBuf.copyImageToBuffer(
        *m_outputImage, vk::ImageLayout::eGeneral,
        *stagingBuf.buffer, region
    );

    cmdBuf.end();

    auto computeQueue = m_device.getQueue(m_computeQueueFamily, 0);
    vk::SubmitInfo submitInfo({}, {}, *cmdBuf);
    computeQueue.submit(submitInfo, nullptr);
    computeQueue.waitIdle();

    // Read back directly: both swapchain and output image now use RGBA format.
    void* mapped = stagingBuf.memory.mapMemory(0, imgSize);

    // Write PNG via stb_image_write (RGBA, 4 components)
    int stride = static_cast<int>(m_config.width) * 4;
    int result = stbi_write_png(
        path.c_str(),
        static_cast<int>(m_config.width),
        static_cast<int>(m_config.height),
        4, mapped, stride
    );
    stagingBuf.memory.unmapMemory();

    if (result == 0) {
        std::cerr << "Failed to write output image: " << path << std::endl;
    } else {
        std::cout << "Saved output to: " << path << std::endl;
    }
}
