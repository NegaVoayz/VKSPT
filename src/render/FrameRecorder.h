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
                bool                              isFirstFrame,
                bool                              showStats = true,
                float                             fps = 0.0f);

    /// Submit the recorded command buffer and present.
    void submit(vk::CommandBuffer  cb,
                uint32_t           frameIdx,
                uint32_t           imageIndex,
                vk::Semaphore      imageAvailable,
                vk::Semaphore      renderFinished,
                vk::Fence          inFlightFence,
                bool               isFirstFrame);

    /// Initialize photon batch state and bind all descriptors.
    /// Must be called once before trySubmitPhotonBatch().
    void setupPhotons(const AccelerationStructure& as,
                      RayTracingPipeline& pipeline,
                      vk::ImageView              outputView,
                      vk::ImageView              normalView,
                      vk::ImageView              depthView,
                      vk::Buffer                 accumBuffer,
                      vk::DeviceSize             accumSize);

    /// If the previous batch fence is signaled and more lights remain,
    /// record and submit the next photon batch asynchronously.
    void trySubmitPhotonBatch(const AccelerationStructure& as,
                              RayTracingPipeline& pipeline);

    bool isPhotonDone() const { return m_photonDone; }

    uint64_t frameCount()      const { return m_frameCount; }
    float    lastGpuMs()       const { return m_lastGpuMs; }

private:
    void transitionImages(vk::CommandBuffer cb, bool firstFrame);
    void dispatchTrace(vk::CommandBuffer            cb,
                       uint32_t                     f,
                       const AccelerationStructure& as,
                       RayTracingPipeline&          pipeline,
                       const CameraParams&          camera,
                       int                          accumFrameCount,
                       float                        fps);
    void dispatchPhotonTrace(vk::CommandBuffer            cb,
                              uint32_t                     f,
                              const AccelerationStructure& as,
                              RayTracingPipeline&          pipeline);
    void dispatchPhotonTraceBatch(vk::CommandBuffer            cb,
                                   uint32_t                     f,
                                   const AccelerationStructure& as,
                                   RayTracingPipeline&          pipeline,
                                   int                           batchIndex,
                                   int                           photonsPerBatch,
                                   int                           totalBatches);
    void dispatchHashCount(vk::CommandBuffer            cb,
                           uint32_t                     f,
                           RayTracingPipeline&          pipeline,
                           int                          photonCount);
    void dispatchHashScan(vk::CommandBuffer            cb,
                          uint32_t                     f,
                          RayTracingPipeline&          pipeline);
    void dispatchHashScatter(vk::CommandBuffer            cb,
                             uint32_t                     f,
                             RayTracingPipeline&          pipeline,
                             int                          photonCount);
    void dispatchHashAggregate(vk::CommandBuffer            cb,
                               uint32_t                     f,
                               RayTracingPipeline&          pipeline);
    void dispatchHashGather(vk::CommandBuffer            cb,
                            uint32_t                     f,
                            RayTracingPipeline&          pipeline);
    void recordPhotonBatch(vk::CommandBuffer            cb,
                            const AccelerationStructure& as,
                            RayTracingPipeline&          pipeline,
                            int                          lightIndex,
                            int                          photonsPerBatch,
                            int                          totalBatches);
    void dispatchStatsOverlay(vk::CommandBuffer    cb,
                              uint32_t             f,
                              RayTracingPipeline&  pipeline);
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
    float     m_lastGpuMs = 0.0f;

    // ---- Async photon batch state ----
    vk::raii::CommandPool   m_photonCmdPool = nullptr;
    vk::raii::CommandBuffer m_photonCmdBuf  = nullptr;
    vk::raii::Fence         m_photonFence;
    vk::ImageView  m_photonOutView  = nullptr;
    vk::ImageView  m_photonNrmView  = nullptr;
    vk::ImageView  m_photonDepView  = nullptr;
    vk::Buffer     m_photonAccumBuf = nullptr;
    vk::DeviceSize m_photonAccumSz  = 0;
    int  m_nextLightIndex   = 0;
    int  m_activeLightCount = 0;
    int  m_perLight         = 0;
    bool m_photonDone       = true;

    int m_photonCount      = 524288;
    int m_photonMaxBounces = 12;
    float m_minNeighborPhotons = 1.0f;  // min G-weighted avg neighbor photon count 
    float m_maxGatherRadius  = 0.20f;  // spatial search radius = 10× cellSize
    float m_hashCellSize     = 0.02f;
};
