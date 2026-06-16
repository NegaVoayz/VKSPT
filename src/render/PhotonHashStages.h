#pragma once

#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>

/// Photon trace dispatch parameters — mirror PhotonRecorder config fields.
struct PhotonTraceConfig {
    int   photonMaxBounces   = 12;
    float minNeighborPhotons = 1.0f;
    float maxGatherRadius    = 0.20f;
    float hashCellSize       = 0.02f;
    int   passCount          = 0;
};

void DispatchPhotonTraceBatch(
    vk::CommandBuffer cb,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline,
    const PhotonTraceConfig& config,
    int batchIndex, int photonsPerBatch, int totalBatches);

void DispatchHashCount(
    vk::CommandBuffer cb,
    RayTracingPipeline& pipeline,
    int photonCount);

void DispatchHashScan(
    vk::CommandBuffer cb,
    RayTracingPipeline& pipeline);

void DispatchHashScatter(
    vk::CommandBuffer cb,
    RayTracingPipeline& pipeline,
    int photonCount);

void DispatchHashAggregate(
    vk::CommandBuffer cb,
    RayTracingPipeline& pipeline);

void DispatchHashGather(
    vk::CommandBuffer cb,
    RayTracingPipeline& pipeline);

void RecordPhotonHashStages(
    vk::CommandBuffer cb,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline,
    int photonsPerBatch);
