#include "render/FramePresent.h"
#include "render/FrameRecorder.h"
#include "core/Log.h"

#include <cstdint>

static constexpr uint32_t kTimestampsPerFrame = 2;

void DispatchFrameTrace(
    vk::CommandBuffer cb, uint32_t f,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    const CameraParams& camera, int accumFrame, float fps,
    float lastGpuMs, const vk::raii::QueryPool* tsPool, bool hasTs,
    uint32_t width, uint32_t height)
{
    FrameRecorder::TracePushConstant pc{};
    pc.camOrigin[0]=camera.origin[0]; pc.camOrigin[1]=camera.origin[1];
    pc.camOrigin[2]=camera.origin[2];
    pc.camU[0]=camera.camU[0]; pc.camU[1]=camera.camU[1];
    pc.camU[2]=camera.camU[2];
    pc.camV[0]=camera.camV[0]; pc.camV[1]=camera.camV[1];
    pc.camV[2]=camera.camV[2];
    pc.camW[0]=camera.camW[0]; pc.camW[1]=camera.camW[1];
    pc.camW[2]=camera.camW[2];
    pc.samplesPerPixel=4; pc.maxBounces=24;
    pc.materialCount=static_cast<int>(as.getMaterialCount());
    pc.fovTan=0.57735f; pc.splitMult=0.25f;
    pc.forceSplitWidth=0.025f; pc.scatterSamples=1;
    pc.mergeThreshold=0.999f; pc.frameIndex=accumFrame;
    pc.diffuseStrength=as.getDiffuseStrength();
    pc.specularStrength=as.getSpecularStrength();
    pc.numLights=static_cast<int>(as.getLightCount());
    pc.minSplitNm=20.0f;
    pc.passType = 0;
    pc.minNeighborPhotons = 1.0f;
    pc.maxGatherRadius = 0.20f;
    pc.hashCellSize = 0.02f;
    pc.photonCount = 524288;
    pc.photonMaxBounces = 12;
    pc.batchIndex = 0;
    pc.batchCount = 1;
    pc.lightIntensityPC = 0;
    pc.lightColorPC[0] = 0; pc.lightColorPC[1] = 0; pc.lightColorPC[2] = 0;
    pc.passCount = 0;
    float gpuFps = lastGpuMs > 0.0f ? 1000.0f / lastGpuMs : fps;
    pc.fps = gpuFps;
    pc.gpuMilliseconds = lastGpuMs;

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.pushConstants<FrameRecorder::TracePushConstant>(pipeline.Desc().PipelineLayout(),
        vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eClosestHitKHR |
            vk::ShaderStageFlagBits::eMissKHR |
            vk::ShaderStageFlagBits::eCompute, 0, pc);

    if (hasTs) {
        cb.resetQueryPool(*tsPool, f * kTimestampsPerFrame, kTimestampsPerFrame);
        cb.writeTimestamp(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                          *tsPool, f * kTimestampsPerFrame);
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdTraceRaysKHR(
        static_cast<VkCommandBuffer>(cb),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.RaygenRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.MissRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.HitRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.CallableRegion()),
        width, height, 1);
}

void TransitionFrameImages(
    vk::CommandBuffer cb, vk::Image out, vk::Image nrm, vk::Image dep, bool first)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    if (first) {
        for (auto img : {out, nrm, dep}) {
            vk::ImageMemoryBarrier b({}, vk::AccessFlagBits::eShaderWrite,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, sub);
            cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, b);
        }
    } else {
        vk::ImageMemoryBarrier b(vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderWrite,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, out, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, b);
    }
}

void CopyOutputToSwapchain(
    vk::CommandBuffer cb, const std::vector<vk::Image>& scImages,
    vk::Image out, uint32_t imageIndex, uint32_t w, uint32_t h)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

    {
        vk::ImageMemoryBarrier ob(vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, out, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, ob);
    }

    vk::Image sw = scImages[imageIndex];

    {
        vk::ImageMemoryBarrier td({}, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, sw, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, td);
    }

    vk::ImageCopy region(
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
        {0,0,0},
        vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
        {0,0,0}, {w, h, 1});
    cb.copyImage(out, vk::ImageLayout::eGeneral,
                 sw, vk::ImageLayout::eTransferDstOptimal, region);

    {
        vk::ImageMemoryBarrier pr(vk::AccessFlagBits::eTransferWrite, {},
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, sw, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, pr);
    }
}

void SubmitAndPresent(
    const vk::raii::Device& device, const vk::raii::SwapchainKHR& sc,
    vk::CommandBuffer cb, uint32_t f, uint32_t imageIndex,
    vk::Semaphore imageAvail, vk::Semaphore renderDone,
    vk::Fence fence, bool first,
    uint32_t cqFamily, uint32_t pqFamily,
    const vk::raii::QueryPool* tsPool, float tsPeriod, bool hasTs,
    uint64_t& frameCount, float& lastGpuMs)
{
    vk::PipelineStageFlags ws = vk::PipelineStageFlagBits::eRayTracingShaderKHR;
    device.getQueue(cqFamily, 0)
        .submit(vk::SubmitInfo(imageAvail, ws, cb, renderDone), fence);
    device.getQueue(pqFamily, 0)
        .presentKHR(vk::PresentInfoKHR(renderDone, *sc, imageIndex));

    if (first)
        device.waitForFences(fence, true, UINT64_MAX);

    if (hasTs && !first) {
        uint64_t r[2];
        auto res = static_cast<vk::Device>(device).getQueryPoolResults(
            *tsPool, f * kTimestampsPerFrame, kTimestampsPerFrame,
            sizeof(r), r, sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (res == vk::Result::eSuccess && r[0] > 0 && r[1] > r[0]) {
            float ms = float(r[1] - r[0]) * tsPeriod / 1e6f;
            lastGpuMs = ms;
            frameCount++;
            if (frameCount % 60 == 0)
                Log::info("[GPU] frame {} compute time: {} ms", frameCount, ms);
        }
    }
}
