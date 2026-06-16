#pragma once

#include "render/CameraParams.h"
#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <vector>

/// Records all per-frame GPU commands: barriers, trace dispatch, denoise
/// dispatch, output→swapchain copy, and present transitions.
/// Also handles submission and timestamp readback.
class FrameRecorder {
public:
    /// Push constant layout matching shaders/rt/payload.slang PushConstants.
    struct TracePushConstant {
        float camOrigin[3]; float pad0;
        float camU[3];      float pad1;
        float camV[3];      float pad2;
        float camW[3];      float pad3;
        int   samplesPerPixel, maxBounces, materialCount;
        float fovTan, splitMult, forceSplitWidth;
        int   scatterSamples; float mergeThreshold;
        int   frameIndex;
        float diffuseStrength, specularStrength; int numLights;
        float minSplitNm;
        int   passType;
        float minNeighborPhotons;
        float maxGatherRadius;
        float hashCellSize;
        int   photonMaxBounces;
        int   photonCount;
        float fps;
        int   batchIndex;
        int   batchCount;
        float lightIntensityPC;
        float lightColorPC[3];
        int   passCount;
        float padEnd[21];
    };

    FrameRecorder(const vk::raii::Device&        device,
                  const vk::raii::SwapchainKHR&  swapchain,
                  const std::vector<vk::Image>&   swapchainImages,
                  vk::Image                      outputImage,
                  vk::Image                      normalImage,
                  vk::Image                      depthImage,
                  uint32_t                       width,
                  uint32_t                       height,
                  uint32_t                       computeQueueFamily,
                  uint32_t                       presentQueueFamily,
                  const vk::raii::QueryPool*     timestampPool,
                  float                          timestampPeriod,
                  bool                           hasTimestamps);

    /// Record trace + denoise + swapchain-copy into the command buffer.
    void record(vk::CommandBuffer                 commandBuffer,
                uint32_t                          frameIndex,
                uint32_t                          imageIndex,
                const AccelerationStructure&      accelerationStructure,
                RayTracingPipeline&               pipeline,
                const CameraParams&               camera,
                int                               accumFrameCount,
                bool                              isFirstFrame,
                bool                              showStats = true,
                float                             fps = 0.0f);

    /// Submit the recorded command buffer and present.
    void submit(vk::CommandBuffer  commandBuffer,
                uint32_t           frameIndex,
                uint32_t           imageIndex,
                vk::Semaphore      imageAvailable,
                vk::Semaphore      renderFinished,
                vk::Fence          inFlightFence,
                bool               isFirstFrame);

    uint64_t frameCount()      const { return m_frameCount; }
    float    lastGpuMilliseconds() const { return m_lastGpuMilliseconds; }

private:
    void transitionImages(vk::CommandBuffer commandBuffer, bool firstFrame);

    void dispatchTrace(vk::CommandBuffer            commandBuffer,
                       uint32_t                     frameIndex,
                       const AccelerationStructure& accelerationStructure,
                       RayTracingPipeline&          pipeline,
                       const CameraParams&          camera,
                       int                          accumFrameCount,
                       float                        fps);

    void dispatchBlendPhoton(vk::CommandBuffer    commandBuffer,
                             uint32_t             frameIndex,
                             RayTracingPipeline&  pipeline);

    void dispatchStatsOverlay(vk::CommandBuffer    commandBuffer,
                              uint32_t             frameIndex,
                              RayTracingPipeline&  pipeline);

    void denoisePass(vk::CommandBuffer    commandBuffer,
                     uint32_t             frameIndex,
                     RayTracingPipeline&  pipeline);

    void copyOutputToSwapchain(vk::CommandBuffer commandBuffer,
                               uint32_t          imageIndex);

    const vk::raii::Device&        m_device;
    const vk::raii::SwapchainKHR&  m_swapchain;
    const std::vector<vk::Image>&  m_swapchainImages;
    vk::Image m_outputImage, m_normalImage, m_depthImage;
    uint32_t  m_width, m_height;
    uint32_t  m_computeQueueFamily, m_presentQueueFamily;
    const vk::raii::QueryPool*     m_timestampPool;
    float     m_timestampPeriod;
    bool      m_hasTimestamps;
    uint64_t  m_frameCount = 0;
    float     m_lastGpuMilliseconds = 0.0f;
};
