#include "render/Renderer.h"

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
    , m_frameCapture(m_device, m_physDevice, m_computeQueueFamily)
    , m_screenshot(m_device, m_physDevice, m_computeQueueFamily)
{
    // 1. Swapchain — may clamp width/height
    m_swapchain.init(m_device, m_physDevice, surface,
                     m_computeQueueFamily, m_presentQueueFamily,
                     m_config.width, m_config.height);

    // 2. Output + G-buffer
    m_output.init(m_device, m_physDevice,
                  m_config.width, m_config.height);
    m_denoiser.init(m_device, m_physDevice,
                    m_config.width, m_config.height);

    // 3. Cross-frame accumulation
    m_accum.init(m_device, m_physDevice, m_computeQueueFamily,
                 m_config.width, m_config.height);

    // 4. Command pool + buffers
    m_commandPool = vk::raii::CommandPool(m_device,
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         m_computeQueueFamily});
    m_commandBuffers = vk::raii::CommandBuffers(m_device,
        {*m_commandPool, vk::CommandBufferLevel::ePrimary,
         MAX_FRAMES_IN_FLIGHT});

    // 5. Sync primitives
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        m_imageAvailableSem.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    for (uint32_t i = 0; i < m_swapchain.imageCount(); ++i)
        m_renderFinishedSem.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        m_inFlightFences.emplace_back(m_device,
            vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));

    // 6. Frame recorder (needs swapchain/output/denoiser to already exist)
    m_recorder.init(m_device, m_swapchain.handle(),
                    m_swapchain.images(),
                    m_output.handle(),
                    m_denoiser.normalImage(),
                    m_denoiser.depthImage(),
                    m_config.width, m_config.height,
                    m_computeQueueFamily, m_presentQueueFamily,
                    m_hasTimestamps ? &m_timestampPool : nullptr,
                    0.0f, false);

    // 7. Timestamp queries
    auto qProps = physDevice.getQueueFamilyProperties();
    if (qProps[computeQueueFamily].timestampValidBits > 0) {
        m_timestampPool = vk::raii::QueryPool(device,
            {{}, vk::QueryType::eTimestamp,
             MAX_FRAMES_IN_FLIGHT * TIMESTAMPS_PER_FRAME});
        m_timestampPeriod =
            physDevice.getProperties().limits.timestampPeriod;
        m_hasTimestamps = true;
        // Re-init recorder with valid timestamp data
        m_recorder.init(m_device, m_swapchain.handle(),
                        m_swapchain.images(),
                        m_output.handle(),
                        m_denoiser.normalImage(),
                        m_denoiser.depthImage(),
                        m_config.width, m_config.height,
                        m_computeQueueFamily, m_presentQueueFamily,
                        &m_timestampPool,
                        m_timestampPeriod, true);
    }
}

Renderer::~Renderer() { m_device.waitIdle(); }

// -----------------------------------------------------------------------------
// renderFrame
// -----------------------------------------------------------------------------
void Renderer::renderFrame(const AccelerationStructure& as,
                            RayTracingPipeline& pipeline,
                            const CameraParams& camera) {
    uint32_t f = m_currentFrame % MAX_FRAMES_IN_FLIGHT;

    if (m_device.waitForFences(*m_inFlightFences[f], true, UINT64_MAX)
        != vk::Result::eSuccess)
        throw std::runtime_error("Fence wait failed.");
    m_device.resetFences(*m_inFlightFences[f]);

    auto [result, imageIndex] = m_swapchain.handle().acquireNextImage(
        UINT64_MAX, *m_imageAvailableSem[f]);
    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("Failed to acquire swapchain image.");

    // Camera movement → reset accumulation
    if (m_accum.detectChange(camera.origin, camera.camU,
                             camera.camV, camera.camW) ||
        m_accum.frameCount() == 0)
        m_accum.reset();
    m_accum.incFrameCount();

    // Bind all descriptor resources
    pipeline.desc().bindTLAS(f, as.getTLAS());
    pipeline.desc().bindOutputImage(f, m_output.view(), nullptr);
    pipeline.desc().bindMaterialBuffer(f, *as.getMaterialBuffer().buffer,
                                       as.getMaterialBuffer().size);
    pipeline.desc().bindLightBuffer(f, *as.getLightBuffer().buffer,
                                    as.getLightBuffer().size);
    pipeline.desc().bindGeometrySSBOs(
        f, *as.getGeometry().vertexBuf().buffer,
           as.getGeometry().vertexBuf().size,
           *as.getGeometry().indexBuf().buffer,
           as.getGeometry().indexBuf().size,
           *as.getGeometry().rangeBuf().buffer,
           as.getGeometry().rangeBuf().size);
    pipeline.desc().bindEnvMap(f, as.getEnvMap().view(),
                               as.getEnvMap().sampler());
    pipeline.desc().bindNormalSSBO(f, *as.getGeometry().normalBuf().buffer,
                                   as.getGeometry().normalBuf().size);
    pipeline.desc().bindAccumBuffer(f, *m_accum.buffer().buffer,
                                    m_accum.bufSize());
    pipeline.desc().bindNormalImage(f, m_denoiser.normalView());
    pipeline.desc().bindDepthImage(f, m_denoiser.depthView());
    pipeline.desc().bindInstanceNormalBuffer(
        f, *as.getInstanceNormalBuffer().buffer,
           as.getInstanceNormalBuffer().size);

    // Record + submit
    auto& cb = m_commandBuffers[f];
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    m_recorder.record(*cb, f, imageIndex, as, pipeline, camera,
                      m_accum.frameCount(), m_currentFrame == 0);
    cb.end();
    m_recorder.submit(*cb, f, imageIndex,
                      *m_imageAvailableSem[f],
                      *m_renderFinishedSem[imageIndex],
                      *m_inFlightFences[f],
                      m_currentFrame == 0);

    m_currentFrame++;
}

// -----------------------------------------------------------------------------
// Delegating public helpers
// -----------------------------------------------------------------------------
void Renderer::saveOutputPNG(const std::string& path) {
    m_frameCapture.savePNG(path, m_output.handle(),
        m_config.width, m_config.height,
        m_inFlightFences, m_commandPool);
}

void Renderer::captureScreenshot(
    const std::string& path, const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    uint32_t capW, uint32_t capH, uint32_t capFrames)
{
    m_screenshot.capture(
        path, as, pipeline, camera, capW, capH, capFrames,
        m_output.handle(), m_output.view(),
        *m_accum.buffer().buffer, m_accum.bufSize(),
        m_denoiser.normalView(), m_denoiser.depthView());
    m_accum.reset();
}
