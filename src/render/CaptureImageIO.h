#pragma once

#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"
#include "render/CameraParams.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <string>

/// Temporary images for offscreen capture (output + normal + depth).
struct CaptureTempImages {
    vk::raii::Image        image         = nullptr;
    vk::raii::DeviceMemory memory        = nullptr;
    vk::raii::ImageView    view          = nullptr;
    vk::raii::Image        normal        = nullptr;
    vk::raii::DeviceMemory normalMemory  = nullptr;
    vk::raii::ImageView    normalView    = nullptr;
    vk::raii::Image        depth         = nullptr;
    vk::raii::DeviceMemory depthMemory   = nullptr;
    vk::raii::ImageView    depthView     = nullptr;
};

/// Allocate R8G8B8A8Unorm / R16G16B16A16Sfloat / R32Sfloat images.
CaptureTempImages CreateCaptureTempImages(
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physDevice,
    uint32_t                        width,
    uint32_t                        height);

/// Trace + denoise one frame at capture resolution.
void TraceAndDenoiseCapture(
    vk::CommandBuffer              commandBuffer,
    uint32_t                       frameIndex,
    uint32_t                       groupCountX,
    uint32_t                       groupCountY,
    const AccelerationStructure&   as,
    RayTracingPipeline&            pipeline,
    const CameraParams&            camera,
    const CaptureTempImages&       temps);

/// Read back capture image to PNG via staging buffer.
void ReadbackCapturePNG(
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physDevice,
    uint32_t                        queueFamily,
    const std::string&              path,
    vk::Image                       image,
    uint32_t                        width,
    uint32_t                        height);
