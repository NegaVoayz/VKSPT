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

    /// Capture at capW×capH for capFrames frames, save to path.
    /// Pass main-resolution resources for descriptor restore.
    void capture(const std::string& path,
                 const AccelerationStructure& as,
                 RayTracingPipeline& pipeline,
                 const CameraParams& camera,
                 uint32_t capW, uint32_t capH, uint32_t capFrames,
                 vk::Image mainOutput, vk::ImageView mainOutputView,
                 vk::Buffer mainAccumBuf, vk::DeviceSize mainAccumSize,
                 vk::ImageView mainNormalView,
                 vk::ImageView mainDepthView);

private:
    struct TempImages {
        vk::raii::Image        img   = nullptr;
        vk::raii::DeviceMemory mem   = nullptr;
        vk::raii::ImageView    view  = nullptr;
        vk::raii::Image        nrm   = nullptr;
        vk::raii::DeviceMemory nrmMem= nullptr;
        vk::raii::ImageView    nrmView=nullptr;
        vk::raii::Image        dep   = nullptr;
        vk::raii::DeviceMemory depMem= nullptr;
        vk::raii::ImageView    depView=nullptr;
    };
    void renderOneFrame(uint32_t f, uint32_t gx, uint32_t gy,
                        const AccelerationStructure& as,
                        RayTracingPipeline& pipeline,
                        const CameraParams& camera,
                        const TempImages& temps);
    void traceAndDenoise(vk::CommandBuffer cb, uint32_t f,
                         uint32_t gx, uint32_t gy,
                         const AccelerationStructure& as,
                         RayTracingPipeline& pipeline,
                         const CameraParams& camera,
                         const TempImages& temps);
    TempImages createTempImages(uint32_t w, uint32_t h);
    void readbackToPNG(const std::string& path, vk::Image img,
                       uint32_t w, uint32_t h,
                       const vk::raii::Queue& queue);

    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    uint32_t                        m_queueFamily;
};
