#pragma once

#include "core/GPUBuffer.h"
#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"
#include "render/CameraParams.h"
#include "render/CrossFrameAccum.h"
#include "render/Denoiser.h"
#include "render/FrameCapture.h"
#include "render/FrameRecorder.h"
#include "render/OutputImage.h"
#include "render/ScreenshotCapture.h"
#include "render/SwapchainManager.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <string>
#include <vector>

/// Frame orchestration: owns all rendering sub-components.
/// Delegates swapchain/image management, accumulation, denoising, command
/// recording, and capture to focused sub-objects.
class Renderer {
public:
    struct Config { uint32_t width; uint32_t height; };

    Renderer(const vk::raii::Instance&       instance,
             const vk::raii::Device&         device,
             const vk::raii::PhysicalDevice& physDevice,
             const vk::raii::SurfaceKHR&     surface,
             uint32_t                        computeQueueFamily,
             uint32_t                        presentQueueFamily,
             const Config&                   config);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&)                 = delete;
    Renderer& operator=(Renderer&&)      = delete;

    void RenderFrame(const AccelerationStructure& as,
                     RayTracingPipeline& pipeline,
                     const CameraParams& camera = {},
                     bool showStats = true,
                     float fps = 0.0f);

    void SaveOutputPNG(const std::string& path);

    void CaptureScreenshot(const std::string& path,
                           const AccelerationStructure& as,
                           RayTracingPipeline& pipeline,
                           const CameraParams& camera,
                           uint32_t capWidth, uint32_t capHeight,
                           uint32_t capFrames);

    int  getAccumCount() const { return m_accum.FrameCount(); }
    float getLastGpuMs() const { return m_recorder.lastGpuMs(); }

    vk::Extent2D GetExtent() const {
        return {m_config.width, m_config.height}; }

private:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t TIMESTAMPS_PER_FRAME = 2;

    // ---- References (must init before dependents) ----
    const vk::raii::Instance&       m_instance;
    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    Config                          m_config;
    uint32_t m_computeQueueFamily = 0;
    uint32_t m_presentQueueFamily = 0;

    // ---- Owned sub-objects (init in ctor body) ----
    SwapchainManager m_swapchain;
    OutputImage      m_output;
    Denoiser         m_denoiser;
    CrossFrameAccum  m_accum;
    FrameCapture     m_frameCapture;
    ScreenshotCapture m_screenshot;
    FrameRecorder    m_recorder;

    uint32_t m_currentFrame = 0;

    // ---- Command & sync ----
    vk::raii::CommandPool                m_commandPool   = nullptr;
    std::vector<vk::raii::CommandBuffer> m_commandBuffers;
    std::vector<vk::raii::Semaphore>     m_imageAvailableSem;
    std::vector<vk::raii::Semaphore>     m_renderFinishedSem;
    std::vector<vk::raii::Fence>         m_inFlightFences;

    // ---- Timestamps ----
    vk::raii::QueryPool m_timestampPool   = nullptr;
    float               m_timestampPeriod = 1.0f;
    bool                m_hasTimestamps   = false;
};
