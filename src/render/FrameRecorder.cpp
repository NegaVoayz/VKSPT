#include "render/FrameRecorder.h"
#include "ray/DescriptorManager.h"
#include "core/Log.h"

static constexpr uint32_t TIMESTAMPS_PER_FRAME = 2;

FrameRecorder::FrameRecorder(
    const vk::raii::Device&         device,
    const vk::raii::SwapchainKHR&   swapchain,
    const std::vector<vk::Image>&   swapchainImages,
    vk::Image out, vk::Image nrm, vk::Image dep,
    uint32_t width, uint32_t height, uint32_t computeQueueFamily, uint32_t presentQueueFamily,
    const vk::raii::QueryPool* timestampPool, float timestampPeriod, bool hasTimestamps)
    : m_device(device)
    , m_swapchain(swapchain)
    , m_swapchainImages(swapchainImages)
    , m_outputImage(out)
    , m_normalImage(nrm)
    , m_depthImage(dep)
    , m_width(width)
    , m_height(height)
    , m_computeQueueFamily(computeQueueFamily)
    , m_presentQueueFamily(presentQueueFamily)
    , m_timestampPool(timestampPool)
    , m_timestampPeriod(timestampPeriod)
    , m_hasTimestamps(hasTimestamps)
{}

void FrameRecorder::record(
    vk::CommandBuffer cb, uint32_t f, uint32_t imageIndex,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    const CameraParams& camera, int accumFrame, bool first, bool showStats, float fps)
{
    transitionImages(cb, first);

    cb.fillBuffer(*as.getRayStats().Buffer, 0, as.getRayStats().Size, 0);

    if (first) {
        cb.fillBuffer(*as.getGatheredCellData().Buffer, 0,
                       as.getGatheredCellData().Size, 0);
        cb.fillBuffer(*as.getDisplayCellData().Buffer, 0,
                       as.getDisplayCellData().Size, 0);
    }

    {
        std::vector<vk::BufferMemoryBarrier> bArr = {
            {vk::AccessFlagBits::eTransferWrite,
             vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getRayStats().Buffer, 0, as.getRayStats().Size},
        };
        if (first) {
            bArr.push_back({vk::AccessFlagBits::eTransferWrite,
             vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getGatheredCellData().Buffer, 0, as.getGatheredCellData().Size});
            bArr.push_back({vk::AccessFlagBits::eTransferWrite,
             vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getDisplayCellData().Buffer, 0, as.getDisplayCellData().Size});
        }
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR |
                vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, bArr, {});
    }

    dispatchBlendPhoton(cb, f, pipeline);

    {
        vk::BufferMemoryBarrier b(vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getDisplayCellData().Buffer, 0, as.getDisplayCellData().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            {}, {}, b, {});
    }

    dispatchTrace(cb, f, as, pipeline, camera, accumFrame, fps);
    denoisePass(cb, f, pipeline);

    if (showStats) {
        vk::BufferMemoryBarrier rsb(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getRayStats().Buffer, 0, as.getRayStats().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, rsb, {});

        dispatchStatsOverlay(cb, f, pipeline);
    }

    copyOutputToSwapchain(cb, imageIndex);
}

void FrameRecorder::dispatchTrace(
    vk::CommandBuffer cb, uint32_t f,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    const CameraParams& camera, int accumFrame, float fps)
{
    TracePushConstant pc{};
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
    pc.fovTan=0.57735f; pc.splitMult=1.0f;
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
    pc.fps = fps;

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.pushConstants<TracePushConstant>(pipeline.Desc().PipelineLayout(),
        vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eClosestHitKHR |
            vk::ShaderStageFlagBits::eMissKHR |
            vk::ShaderStageFlagBits::eCompute, 0, pc);

    if (m_hasTimestamps) {
        cb.resetQueryPool(*m_timestampPool, f * TIMESTAMPS_PER_FRAME, TIMESTAMPS_PER_FRAME);
        cb.writeTimestamp(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                          *m_timestampPool, f * TIMESTAMPS_PER_FRAME);
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdTraceRaysKHR(
        static_cast<VkCommandBuffer>(cb),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.RaygenRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.MissRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.HitRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.CallableRegion()),
        m_width, m_height, 1);
}

void FrameRecorder::dispatchBlendPhoton(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetBlendPhotonPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.dispatch(2048, 1, 1);   // HASH_TABLE_SIZE / 256
}

void FrameRecorder::transitionImages(vk::CommandBuffer cb, bool first)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    if (first) {
        for (auto img : {m_outputImage, m_normalImage, m_depthImage}) {
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
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_outputImage, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, b);
    }
}

void FrameRecorder::dispatchStatsOverlay(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetStatsOverlayPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.dispatch(35, 23, 1);
}

void FrameRecorder::denoisePass(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageMemoryBarrier gb[3] = {
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_normalImage, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_depthImage, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_outputImage, sub},
    };
    vk::MemoryBarrier mb(vk::AccessFlagBits::eShaderWrite,
                         vk::AccessFlagBits::eShaderRead);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits::eComputeShader, {}, mb, {}, gb);

    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetDenoisePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    uint32_t gx = (m_width + 7) / 8, gy = (m_height + 7) / 8;
    cb.dispatch(gx, gy, 1);

    if (m_hasTimestamps)
        cb.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                          *m_timestampPool, f * TIMESTAMPS_PER_FRAME + 1);
}

void FrameRecorder::copyOutputToSwapchain(
    vk::CommandBuffer cb, uint32_t imageIndex)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

    {
        vk::ImageMemoryBarrier ob(vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_outputImage, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, ob);
    }

    vk::Image sw = m_swapchainImages[imageIndex];

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
        {0,0,0}, {m_width, m_height, 1});
    cb.copyImage(m_outputImage, vk::ImageLayout::eGeneral,
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

void FrameRecorder::submit(
    vk::CommandBuffer cb, uint32_t f, uint32_t imageIndex,
    vk::Semaphore imageAvail, vk::Semaphore renderDone,
    vk::Fence fence, bool first)
{
    vk::PipelineStageFlags ws = vk::PipelineStageFlagBits::eRayTracingShaderKHR;
    m_device.getQueue(m_computeQueueFamily, 0)
        .submit(vk::SubmitInfo(imageAvail, ws, cb, renderDone), fence);
    m_device.getQueue(m_presentQueueFamily, 0)
        .presentKHR(vk::PresentInfoKHR(renderDone, *m_swapchain, imageIndex));

    if (first)
        m_device.waitForFences(fence, true, UINT64_MAX);

    if (m_hasTimestamps && !first) {
        uint64_t r[2];
        auto res = static_cast<vk::Device>(m_device).getQueryPoolResults(
            *m_timestampPool, f * TIMESTAMPS_PER_FRAME, TIMESTAMPS_PER_FRAME,
            sizeof(r), r, sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (res == vk::Result::eSuccess && r[0] > 0 && r[1] > r[0]) {
            float ms = float(r[1] - r[0]) * m_timestampPeriod / 1e6f;
            m_lastGpuMilliseconds = ms;
            m_frameCount++;
            if (m_frameCount % 60 == 0)
                Log::info("[GPU] frame {} compute time: {} ms",
                          m_frameCount, ms);
        }
    }
}
