#pragma once

#include "render/CameraParams.h"
#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>

/// Record frame trace dispatch (sets push constants, binds pipeline, calls vkCmdTraceRays).
void DispatchFrameTrace(
    vk::CommandBuffer commandBuffer,
    uint32_t frameIndex,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline,
    const CameraParams& camera,
    int accumFrameCount,
    float fps,
    float lastGpuMilliseconds,
    const vk::raii::QueryPool* timestampPool,
    bool hasTimestamps,
    uint32_t width, uint32_t height);

/// Transition output/normal/depth images for compute shader access.
void TransitionFrameImages(
    vk::CommandBuffer commandBuffer,
    vk::Image outputImage,
    vk::Image normalImage,
    vk::Image depthImage,
    bool isFirstFrame);

/// Copy compute output to swapchain image with barrier transitions.
void CopyOutputToSwapchain(
    vk::CommandBuffer commandBuffer,
    const std::vector<vk::Image>& swapchainImages,
    vk::Image outputImage,
    uint32_t imageIndex,
    uint32_t width, uint32_t height);

/// Submit work to compute queue and present to swapchain.
void SubmitAndPresent(
    const vk::raii::Device& device,
    const vk::raii::SwapchainKHR& swapchain,
    vk::CommandBuffer commandBuffer,
    uint32_t frameIndex,
    uint32_t imageIndex,
    vk::Semaphore imageAvailable,
    vk::Semaphore renderFinished,
    vk::Fence inFlightFence,
    bool isFirstFrame,
    uint32_t computeQueueFamily,
    uint32_t presentQueueFamily,
    const vk::raii::QueryPool* timestampPool,
    float timestampPeriod,
    bool hasTimestamps,
    uint64_t& frameCount,
    float& lastGpuMilliseconds);
