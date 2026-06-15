#pragma once

#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"
#include "render/CameraParams.h"
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <string>

/// High-resolution offscreen screenshot: render N frames at target resolution,
/// accumulate, denoise, and save to PNG.
class ScreenshotCapture {
public:
    ScreenshotCapture(const vk::raii::Device&         device,
                      const vk::raii::PhysicalDevice& physDevice,
                      uint32_t                        queueFamily);

    /// Capture at targetWidth×targetHeight for frameCount frames, save to path.
    /// Pass main-resolution resources for descriptor restore.
    void capture(const std::string& path,
                 const AccelerationStructure& as,
                 RayTracingPipeline& pipeline,
                 const CameraParams& camera,
                 uint32_t targetWidth, uint32_t targetHeight, uint32_t frameCount,
                 vk::Image mainOutput, vk::ImageView mainOutputView,
                 vk::Buffer mainAccumBuffer, vk::DeviceSize mainAccumSize,
                 vk::ImageView mainNormalView,
                 vk::ImageView mainDepthView);

private:
    struct TempImages {
        vk::raii::Image        image      = nullptr;
        vk::raii::DeviceMemory memory     = nullptr;
        vk::raii::ImageView    view       = nullptr;
        vk::raii::Image        normal     = nullptr;
        vk::raii::DeviceMemory normalMemory = nullptr;
        vk::raii::ImageView    normalView = nullptr;
        vk::raii::Image        depth     = nullptr;
        vk::raii::DeviceMemory depthMemory = nullptr;
        vk::raii::ImageView    depthView = nullptr;
    };

    TempImages createTempImages(uint32_t width, uint32_t height);
    void createOneTempImage(uint32_t width, uint32_t height, vk::Format format,
                            vk::raii::Image& outImage,
                            vk::raii::DeviceMemory& outMemory,
                            vk::raii::ImageView& outView);

    void renderOneFrame(uint32_t frameIndex, uint32_t groupCountX, uint32_t groupCountY,
                        const AccelerationStructure& as,
                        RayTracingPipeline& pipeline,
                        const CameraParams& camera,
                        const TempImages& temps);

    void traceAndDenoise(vk::CommandBuffer commandBuffer, uint32_t frameIndex,
                         uint32_t groupCountX, uint32_t groupCountY,
                         const AccelerationStructure& as,
                         RayTracingPipeline& pipeline,
                         const CameraParams& camera,
                         const TempImages& temps);

    void readbackToPNG(const std::string& path, vk::Image image,
                       uint32_t width, uint32_t height,
                       const vk::raii::Queue& queue);

    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    uint32_t                        m_queueFamily;
};
