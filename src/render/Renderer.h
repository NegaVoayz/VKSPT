#pragma once

#include "ray/AccelerationStructure.h"
#include "core/GPUBuffer.h"
#include "ray/RayTracingPipeline.h"
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <string>
#include <vector>

/// Frame orchestration: swapchain, compute dispatch, present, and PPM output.
/// Owns the swapchain, storage image (compute target), and command buffers.
class Renderer {
public:
    struct Config {
        uint32_t width;
        uint32_t height;
    };

    /// Camera parameters passed from the application each frame.
    struct CameraParams {
        float origin[3];
        float camU[3];
        float camV[3];
        float camW[3];
    };

    Renderer(
        const vk::raii::Instance&       instance,
        const vk::raii::Device&         device,
        const vk::raii::PhysicalDevice& physDevice,
        const vk::raii::SurfaceKHR&     surface,
        uint32_t                        computeQueueFamily,
        uint32_t                        presentQueueFamily,
        const Config&                   config
    );
    ~Renderer();

    // Non-copyable, non-movable
    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&)                 = delete;
    Renderer& operator=(Renderer&&)      = delete;

    /// Dispatch one frame: compute → copy to swapchain → present.
    void renderFrame(const AccelerationStructure& as, RayTracingPipeline& pipeline,
                     const CameraParams& camera = {});

    /// Save the current compute output as a PNG file.
    void saveOutputPNG(const std::string& path);

    /// Capture a high-resolution screenshot by rendering offscreen at the given
    /// resolution for capFrames frames, then saving to path. Blocks until done.
    void captureScreenshot(const std::string& path,
                           const AccelerationStructure& as,
                           RayTracingPipeline& pipeline,
                           const CameraParams& camera,
                           uint32_t capWidth, uint32_t capHeight,
                           uint32_t capFrames);

    vk::Extent2D getExtent() const { return {m_config.width, m_config.height}; }

private:
    void createSwapchain(const vk::raii::SurfaceKHR& surface);
    void createOutputImage();
    void createCommandBuffers();
    void createSyncObjects();

    void transitionImageLayout(
        vk::CommandBuffer cmd,
        vk::Image         image,
        vk::ImageLayout   oldLayout,
        vk::ImageLayout   newLayout
    );

    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    const vk::raii::Instance&       m_instance;
    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    Config                          m_config;

    uint32_t m_computeQueueFamily = 0;
    uint32_t m_presentQueueFamily = 0;
    uint32_t m_swapchainImageCount = 0;
    uint32_t m_currentFrame = 0;

    // Swapchain
    vk::raii::SwapchainKHR             m_swapchain        = nullptr;
    std::vector<vk::Image>             m_swapchainImages;
    std::vector<vk::raii::ImageView>   m_swapchainViews;

    // Output storage image (compute shader target) — stays in GENERAL layout
    vk::raii::Image           m_outputImage         = nullptr;
    vk::raii::DeviceMemory    m_outputMemory        = nullptr;
    vk::raii::ImageView       m_outputView          = nullptr;

    // Command pool & buffers (one per frame-in-flight)
    vk::raii::CommandPool                m_commandPool         = nullptr;
    std::vector<vk::raii::CommandBuffer> m_commandBuffers;

    // Synchronization primitives
    // imageAvailable: per-frame-in-flight (acquire signals these)
    // renderFinished: per-swapchain-image (submit signals, present waits — must be unique per image)
    // inFlightFences: per-frame-in-flight (CPU-GPU sync)
    std::vector<vk::raii::Semaphore> m_imageAvailableSem;   // MAX_FRAMES_IN_FLIGHT
    std::vector<vk::raii::Semaphore> m_renderFinishedSem;   // swapchainImageCount
    std::vector<vk::raii::Fence>     m_inFlightFences;       // MAX_FRAMES_IN_FLIGHT

    // Timestamp queries for GPU profiling
    static constexpr uint32_t TIMESTAMPS_PER_FRAME = 2;  // start + end
    vk::raii::QueryPool          m_timestampPool = nullptr;
    float                        m_timestampPeriod = 1.0f;  // nanoseconds per tick
    uint64_t                     m_frameCount = 0;
    bool                         m_hasTimestamps = false;

    // Phase 6: Cross-frame accumulation for progressive rendering
    GPUBuffer   m_accumBuffer;        // device-local SSBO, width×height×4 floats (r,g,b,count)
    GPUBuffer   m_accumStaging;       // staging for zero-reset
    CameraParams m_lastCamera{};      // previous frame camera (detect movement → reset)
    int          m_accumFrameCount = 0;

    /// Zero-fill the accumulation buffer via staging copy.
    void resetAccumBuffer();

    // Phase 6.5: G-buffer storage images for denoising
    vk::raii::Image        m_normalImage   = nullptr;
    vk::raii::DeviceMemory m_normalMemory  = nullptr;
    vk::raii::ImageView    m_normalView    = nullptr;
    vk::raii::Image        m_depthImage    = nullptr;
    vk::raii::DeviceMemory m_depthMemory   = nullptr;
    vk::raii::ImageView    m_depthView     = nullptr;

    /// Create G-buffer storage images (rgba16f normals, r32f depth).
    void createGBufferImages();

};
