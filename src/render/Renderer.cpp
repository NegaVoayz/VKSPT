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
    , m_swapchain(m_device, m_physDevice, surface,
                  m_computeQueueFamily, m_presentQueueFamily,
                  m_config.width, m_config.height)
    , m_output(m_device, m_physDevice, m_config.width, m_config.height)
    , m_denoiser(m_device, m_physDevice, m_config.width, m_config.height)
    , m_accum(m_device, m_physDevice, m_computeQueueFamily,
              m_config.width, m_config.height)
    , m_frameCapture(m_device, m_physDevice, m_computeQueueFamily)
    , m_screenshot(m_device, m_physDevice, m_computeQueueFamily)
    , m_recorder(m_device, m_swapchain.Handle(), m_swapchain.Images(),
                 m_output.Handle(), m_denoiser.NormalImage(),
                 m_denoiser.DepthImage(),
                 m_config.width, m_config.height,
                 m_computeQueueFamily, m_presentQueueFamily,
                 nullptr, 0.0f, false)
{
    // Timestamp queries
    auto qProps = physDevice.getQueueFamilyProperties();
    if (qProps[computeQueueFamily].timestampValidBits > 0) {
        m_timestampPool = vk::raii::QueryPool(device,
            {{}, vk::QueryType::eTimestamp,
             MAX_FRAMES_IN_FLIGHT * TIMESTAMPS_PER_FRAME});
        m_timestampPeriod =
            physDevice.getProperties().limits.timestampPeriod;
        m_hasTimestamps = true;
    }

    m_commandPool = vk::raii::CommandPool(m_device,
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         m_computeQueueFamily});
    m_commandBuffers = vk::raii::CommandBuffers(m_device,
        {*m_commandPool, vk::CommandBufferLevel::ePrimary,
         MAX_FRAMES_IN_FLIGHT});

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        m_imageAvailableSem.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    for (uint32_t i = 0; i < m_swapchain.ImageCount(); ++i)
        m_renderFinishedSem.emplace_back(m_device, vk::SemaphoreCreateInfo{});
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        m_inFlightFences.emplace_back(m_device,
            vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
}

Renderer::~Renderer() { m_device.waitIdle(); }

// -----------------------------------------------------------------------------
// renderFrame
// -----------------------------------------------------------------------------
void Renderer::RenderFrame(const AccelerationStructure& as,
                            RayTracingPipeline& pipeline,
                            const CameraParams& camera) {
    uint32_t f = m_currentFrame % MAX_FRAMES_IN_FLIGHT;

    if (m_device.waitForFences(*m_inFlightFences[f], true, UINT64_MAX)
        != vk::Result::eSuccess)
        throw std::runtime_error("Fence wait failed.");
    m_device.resetFences(*m_inFlightFences[f]);

    auto [result, imageIndex] = m_swapchain.Handle().acquireNextImage(
        UINT64_MAX, *m_imageAvailableSem[f]);
    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("Failed to acquire swapchain image.");

    // Camera movement → reset accumulation
    if (m_accum.detectChange(camera.origin, camera.camU,
                             camera.camV, camera.camW) ||
        m_accum.FrameCount() == 0)
        m_accum.reset();
    m_accum.IncFrameCount();

    // Bind all descriptor resources
    pipeline.Desc().BindTLAS(f, as.getTLAS());
    pipeline.Desc().BindOutputImage(f, m_output.View(), nullptr);
    pipeline.Desc().BindMaterialBuffer(f, *as.getMaterialBuffer().Buffer,
                                       as.getMaterialBuffer().Size);
    pipeline.Desc().BindLightBuffer(f, *as.getLightBuffer().Buffer,
                                    as.getLightBuffer().Size);
    pipeline.Desc().BindGeometrySSBOs(
        f, *as.getGeometry().vertexBuf().Buffer,
           as.getGeometry().vertexBuf().Size,
           *as.getGeometry().indexBuf().Buffer,
           as.getGeometry().indexBuf().Size,
           *as.getGeometry().rangeBuf().Buffer,
           as.getGeometry().rangeBuf().Size);
    pipeline.Desc().BindEnvMap(f, as.getEnvMap().view(),
                               as.getEnvMap().sampler());
    pipeline.Desc().BindNormalSSBO(f, *as.getGeometry().normalBuf().Buffer,
                                   as.getGeometry().normalBuf().Size);
    pipeline.Desc().BindAccumBuffer(f, *m_accum.Buffer().Buffer,
                                    m_accum.BufSize());
    pipeline.Desc().BindNormalImage(f, m_denoiser.NormalView());
    pipeline.Desc().BindDepthImage(f, m_denoiser.DepthView());
    pipeline.Desc().BindInstanceNormalBuffer(
        f, *as.getInstanceNormalBuffer().Buffer,
           as.getInstanceNormalBuffer().Size);
    pipeline.Desc().BindPhotonBuffer(
        f, *as.getPhotonBuffer().Buffer, as.getPhotonBuffer().Size);
    pipeline.Desc().BindPhotonCounter(
        f, *as.getPhotonCounter().Buffer, as.getPhotonCounter().Size);

    // Record + submit
    auto& cb = m_commandBuffers[f];
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    m_recorder.record(*cb, f, imageIndex, as, pipeline, camera,
                      m_accum.FrameCount(), m_currentFrame == 0);
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
void Renderer::SaveOutputPNG(const std::string& path) {
    m_frameCapture.savePNG(path, m_output.Handle(),
        m_config.width, m_config.height,
        m_inFlightFences, m_commandPool);
}

void Renderer::CaptureScreenshot(
    const std::string& path, const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    uint32_t capW, uint32_t capH, uint32_t capFrames)
{
    m_screenshot.capture(
        path, as, pipeline, camera, capW, capH, capFrames,
        m_output.Handle(), m_output.View(),
        *m_accum.Buffer().Buffer, m_accum.BufSize(),
        m_denoiser.NormalView(), m_denoiser.DepthView());
    m_accum.reset();
}
