#pragma once

#include "AccelerationStructure.h"
#include "GPUBuffer.h"
#include "RayTracingPipeline.h"
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

    /// Save the current compute output as a PPM file.
    void saveOutputPPM(const std::string& path);

    vk::Extent2D getExtent() const { return {m_config.width, m_config.height}; }

private:
    void createSwapchain(const vk::raii::SurfaceKHR& surface);
    void createOutputImage();
    void createCommandPool();
    void createSyncObjects();

    void transitionImageLayout(
        vk::CommandBuffer cmd,
        vk::Image         image,
        vk::ImageLayout   oldLayout,
        vk::ImageLayout   newLayout
    );

    const vk::raii::Instance&       m_instance;
    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    Config                          m_config;

    uint32_t m_computeQueueFamily = 0;
    uint32_t m_presentQueueFamily = 0;
    uint32_t m_swapchainImageCount = 0;

    // Swapchain
    vk::raii::SwapchainKHR             m_swapchain        = nullptr;
    std::vector<vk::Image>             m_swapchainImages;
    std::vector<vk::raii::ImageView>   m_swapchainViews;

    // Output storage image (compute shader target)
    vk::raii::Image           m_outputImage         = nullptr;
    vk::raii::DeviceMemory    m_outputMemory        = nullptr;
    vk::raii::ImageView       m_outputView          = nullptr;

    // Command pool
    vk::raii::CommandPool     m_commandPool         = nullptr;
    vk::raii::CommandBuffer   m_computeCmdBuf       = nullptr;

    // Synchronization
    vk::raii::Semaphore       m_imageAvailable      = nullptr;
    vk::raii::Semaphore       m_renderFinished      = nullptr;
    vk::raii::Fence           m_inFlightFence       = nullptr;
};
