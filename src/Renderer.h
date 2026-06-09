#pragma once

#include "AccelerationStructure.h"
#include "GPUBuffer.h"
#include "RaySorter.h"
#include "RayTracingPipeline.h"
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <memory>
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
    void renderFrame(const AccelerationStructure& as, RayTracingPipeline& pipeline);

    /// Save the current compute output as a PNG file.
    void saveOutputPNG(const std::string& path);

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

    void initSortedPipeline(RayTracingPipeline& pipeline);

    // Timestamp queries for GPU profiling
    static constexpr uint32_t TIMESTAMPS_PER_FRAME = 2;  // start + end
    vk::raii::QueryPool          m_timestampPool = nullptr;
    float                        m_timestampPeriod = 1.0f;  // nanoseconds per tick
    uint64_t                     m_frameCount = 0;
    bool                         m_hasTimestamps = false;

    // Phase 4: Sorted ray tracing pipeline
    std::unique_ptr<RaySorter>   m_raySorter;
    bool                         m_useSorting = false;  // true = global ray buffer pipeline
};
