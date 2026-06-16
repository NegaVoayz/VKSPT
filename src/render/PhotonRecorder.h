#pragma once

#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>

/// Records photon trace + hash-grid pipeline dispatches for one light batch.
/// Owns the async photon command pool, command buffer, and fence.
class PhotonRecorder {
public:
    PhotonRecorder(const vk::raii::Device& device,
                   uint32_t computeQueueFamily);

    /// Bind all photon descriptors and prepare batch state.
    /// Must be called once before trySubmitPhotonBatch().
    void setupPhotons(const AccelerationStructure& as,
                      RayTracingPipeline& pipeline,
                      vk::ImageView outputView,
                      vk::ImageView normalView,
                      vk::ImageView depthView,
                      vk::Buffer accumBuffer,
                      vk::DeviceSize accumSize);

    /// If previous batch fence signaled and more lights remain,
    /// record + submit next photon batch asynchronously.
    void trySubmitPhotonBatch(const AccelerationStructure& as,
                              RayTracingPipeline& pipeline);

    bool isPhotonDone() const { return m_photonDone; }

private:
    void bindPhotonDescriptors(const AccelerationStructure& as,
                               RayTracingPipeline& pipeline);

    void recordPhotonBatch(vk::CommandBuffer cb,
                           const AccelerationStructure& as,
                           RayTracingPipeline& pipeline,
                           int lightIndex, int photonsPerBatch,
                           int totalBatches);

    void recordHashStages(vk::CommandBuffer cb,
                          const AccelerationStructure& as,
                          RayTracingPipeline& pipeline,
                          int photonsPerBatch);

    void dispatchPhotonTraceBatch(vk::CommandBuffer cb,
                                  const AccelerationStructure& as,
                                  RayTracingPipeline& pipeline,
                                  int batchIndex,
                                  int photonsPerBatch,
                                  int totalBatches);

    void dispatchHashCount(vk::CommandBuffer cb,
                           RayTracingPipeline& pipeline,
                           int photonCount);

    void dispatchHashScan(vk::CommandBuffer cb,
                          RayTracingPipeline& pipeline);

    void dispatchHashScatter(vk::CommandBuffer cb,
                             RayTracingPipeline& pipeline,
                             int photonCount);

    void dispatchHashAggregate(vk::CommandBuffer cb,
                               RayTracingPipeline& pipeline);

    void dispatchHashGather(vk::CommandBuffer cb,
                            RayTracingPipeline& pipeline);

    const vk::raii::Device& m_device;
    uint32_t m_computeQueueFamily;

    vk::raii::CommandPool   m_photonCommandPool = nullptr;
    vk::raii::CommandBuffer m_photonCommandBuffer = nullptr;
    vk::raii::Fence         m_photonFence;
    vk::ImageView  m_photonOutputView = nullptr;
    vk::ImageView  m_photonNormalView = nullptr;
    vk::ImageView  m_photonDepthView = nullptr;
    vk::Buffer     m_photonAccumBuffer = nullptr;
    vk::DeviceSize m_photonAccumSize = 0;
    int  m_nextLightIndex   = 0;
    int  m_activeLightCount = 0;
    int  m_perLight         = 0;
    int  m_passCount        = 0;
    bool m_photonDone       = true;

    int   m_photonCount        = 524288;
    int   m_photonMaxBounces   = 12;
    float m_minNeighborPhotons = 1.0f;
    float m_maxGatherRadius    = 0.20f;
    float m_hashCellSize       = 0.02f;
};
