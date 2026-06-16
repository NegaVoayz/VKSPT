#include "render/PhotonHashStages.h"
#include "render/FrameRecorder.h"
#include "ray/DescriptorManager.h"

#include <vector>

namespace {

void barrierComputeToCompute(vk::CommandBuffer cb, vk::Buffer buf, vk::DeviceSize size)
{
    vk::BufferMemoryBarrier b(vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        buf, 0, size);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader, {}, {}, b, {});
}

void barrierComputeRead(vk::CommandBuffer cb, vk::Buffer buf, vk::DeviceSize size)
{
    vk::BufferMemoryBarrier b(vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        buf, 0, size);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader, {}, {}, b, {});
}

} // namespace

void DispatchPhotonTraceBatch(
    vk::CommandBuffer cb,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline,
    const PhotonTraceConfig& config,
    int batchIndex, int photonsPerBatch, int totalBatches)
{
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    FrameRecorder::TracePushConstant pc{};

    pc.samplesPerPixel = 1; pc.maxBounces = config.photonMaxBounces;
    pc.materialCount = static_cast<int>(as.getMaterialCount());
    pc.fovTan = 0.57735f; pc.splitMult = 0.25f;
    pc.forceSplitWidth = 0.002f; pc.scatterSamples = 1;
    pc.mergeThreshold = 0.999f; pc.frameIndex = 0;
    pc.diffuseStrength = as.getDiffuseStrength();
    pc.specularStrength = as.getSpecularStrength();
    pc.numLights = static_cast<int>(as.getLightCount());
    pc.minSplitNm = 20.0f;
    pc.passType = 1;
    pc.minNeighborPhotons = config.minNeighborPhotons;
    pc.maxGatherRadius = config.maxGatherRadius;
    pc.hashCellSize = config.hashCellSize;
    pc.photonMaxBounces = config.photonMaxBounces;
    pc.photonCount = photonsPerBatch;
    pc.batchIndex = batchIndex;
    pc.batchCount = totalBatches;
    auto& lightCpu = as.getLightsCPU()[batchIndex];
    pc.lightIntensityPC = lightCpu.color_intensity[3];
    pc.lightColorPC[0] = lightCpu.color_intensity[0];
    pc.lightColorPC[1] = lightCpu.color_intensity[1];
    pc.lightColorPC[2] = lightCpu.color_intensity[2];
    pc.passCount = config.passCount + 1;
    pc.gpuMilliseconds = 0.0f;

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(ps), nullptr);
    cb.pushConstants<FrameRecorder::TracePushConstant>(pipeline.Desc().PipelineLayout(),
        vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eClosestHitKHR |
            vk::ShaderStageFlagBits::eMissKHR |
            vk::ShaderStageFlagBits::eCompute, 0, pc);

    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdTraceRaysKHR(
        static_cast<VkCommandBuffer>(cb),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.PhotonRaygenRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.MissRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.HitRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.CallableRegion()),
        photonsPerBatch, 1, 1);
}

void DispatchHashCount(
    vk::CommandBuffer cb, RayTracingPipeline& pipeline, int photonCount)
{
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashCountPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(ps), nullptr);
    cb.dispatch(uint32_t(photonCount) * 16 / 256, 1, 1);
}

void DispatchHashScan(
    vk::CommandBuffer cb, RayTracingPipeline& pipeline)
{
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashScanPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(ps), nullptr);
    cb.dispatch(1, 1, 1);
}

void DispatchHashScatter(
    vk::CommandBuffer cb, RayTracingPipeline& pipeline, int photonCount)
{
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashScatterPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(ps), nullptr);
    cb.dispatch(uint32_t(photonCount) * 16 / 256, 1, 1);
}

void DispatchHashAggregate(
    vk::CommandBuffer cb, RayTracingPipeline& pipeline)
{
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashAggregatePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(ps), nullptr);
    cb.dispatch(2048, 1, 1);
}

void DispatchHashGather(
    vk::CommandBuffer cb, RayTracingPipeline& pipeline)
{
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashGatherPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(ps), nullptr);
    cb.dispatch(2048, 1, 1);
}

void RecordPhotonHashStages(
    vk::CommandBuffer cb,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline,
    int photonsPerBatch)
{
    // count
    DispatchHashCount(cb, pipeline, photonsPerBatch);
    barrierComputeToCompute(cb, *as.getHashCellData().Buffer, as.getHashCellData().Size);

    // scan
    DispatchHashScan(cb, pipeline);
    barrierComputeToCompute(cb, *as.getHashCellData().Buffer, as.getHashCellData().Size);

    // scatter
    DispatchHashScatter(cb, pipeline, photonsPerBatch);
    {
        vk::Buffer hashBuf = *as.getHashCellData().Buffer;
        vk::Buffer idxBuf = *as.getSortedPhotonIndices().Buffer;
        vk::DeviceSize hashSz = as.getHashCellData().Size;
        vk::DeviceSize idxSz = as.getSortedPhotonIndices().Size;
        vk::BufferMemoryBarrier bArr[2] = {
            {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, hashBuf, 0, hashSz},
            {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, idxBuf, 0, idxSz},
        };
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader, {}, {}, bArr, {});
    }

    // aggregate
    DispatchHashAggregate(cb, pipeline);
    barrierComputeRead(cb, *as.getCellPhotonData().Buffer, as.getCellPhotonData().Size);

    // gather → RT
    DispatchHashGather(cb, pipeline);
    {
        vk::BufferMemoryBarrier b(vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getGatheredCellData().Buffer, 0, as.getGatheredCellData().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, {}, b, {});
    }
}
