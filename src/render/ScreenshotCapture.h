#pragma once

#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"
#include "render/CameraParams.h"
#include "render/CaptureImageIO.h"

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
    void renderOneFrame(uint32_t frameIndex, uint32_t groupCountX, uint32_t groupCountY,
                        const AccelerationStructure& as,
                        RayTracingPipeline& pipeline,
                        const CameraParams& camera,
                        const CaptureTempImages& temps);

    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    uint32_t                        m_queueFamily;
};
