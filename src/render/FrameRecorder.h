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
    FrameRecorder(const vk::raii::Device&        device,
                  const vk::raii::SwapchainKHR&  swapchain,
                  const std::vector<vk::Image>&   swapchainImages,
                  vk::Image                      outputImage,
                  vk::Image                      normalImage,
                  vk::Image                      depthImage,
                  uint32_t                       width,
                  uint32_t                       height,
                  uint32_t                       computeQf,
                  uint32_t                       presentQf,
                  const vk::raii::QueryPool*     timestampPool,
                  float                          timestampPeriod,
                  bool                           hasTimestamps);

    /// Record trace + denoise + swapchain-copy into the command buffer.
    void record(vk::CommandBuffer                 cb,
                uint32_t                          frameIdx,
                uint32_t                          imageIndex,
                const AccelerationStructure&      as,
                RayTracingPipeline&               pipeline,
                const CameraParams&               camera,
                int                               accumFrameCount,
                bool                              isFirstFrame);

    /// Submit the recorded command buffer and present.
    void submit(vk::CommandBuffer  cb,
                uint32_t           frameIdx,
                uint32_t           imageIndex,
                vk::Semaphore      imageAvailable,
                vk::Semaphore      renderFinished,
                vk::Fence          inFlightFence,
                bool               isFirstFrame);

    uint64_t frameCount() const { return m_frameCount; }

private:
    void transitionImages(vk::CommandBuffer cb, bool firstFrame);
    void dispatchTrace(vk::CommandBuffer            cb,
                       uint32_t                     f,
                       const AccelerationStructure& as,
                       RayTracingPipeline&          pipeline,
                       const CameraParams&          camera,
                       int                          accumFrameCount);
    void dispatchPhotonTrace(vk::CommandBuffer            cb,
                              uint32_t                     f,
                              const AccelerationStructure& as,
                              RayTracingPipeline&          pipeline);
    void dispatchHashCount(vk::CommandBuffer            cb,
                           uint32_t                     f,
                           RayTracingPipeline&          pipeline);
    void dispatchHashScan(vk::CommandBuffer            cb,
                          uint32_t                     f,
                          RayTracingPipeline&          pipeline);
    void dispatchHashScatter(vk::CommandBuffer            cb,
                             uint32_t                     f,
                             RayTracingPipeline&          pipeline);
    void denoisePass(vk::CommandBuffer    cb,
                     uint32_t             f,
                     RayTracingPipeline&  pipeline);
    void copyOutputToSwapchain(vk::CommandBuffer cb,
                               uint32_t          imageIndex);

    const vk::raii::Device&        m_device;
    const vk::raii::SwapchainKHR&  m_swapchain;
    const std::vector<vk::Image>&  m_swapImages;
    vk::Image m_outImg, m_nrmImg, m_depImg;
    uint32_t  m_w, m_h;
    uint32_t  m_cqf, m_pqf;
    const vk::raii::QueryPool*     m_tsPool;  // nullable — optional query pool
    float     m_tsPeriod;
    bool      m_hasTS;
    uint64_t  m_frameCount = 0;
    int m_photonCount = 65536;
    int m_photonMaxBounces = 12;
    float m_gatherRadius = 0.02f;
};
