#include "render/FrameRecorder.h"
#include "render/FramePresent.h"
#include "ray/DescriptorManager.h"

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
    TransitionFrameImages(cb, m_outputImage, m_normalImage, m_depthImage, first);

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

    DispatchFrameTrace(cb, f, as, pipeline, camera, accumFrame, fps,
                       m_lastGpuMilliseconds, m_timestampPool,
                       m_hasTimestamps, m_width, m_height);
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

    CopyOutputToSwapchain(cb, m_swapchainImages, m_outputImage, imageIndex,
                          m_width, m_height);
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

void FrameRecorder::submit(
    vk::CommandBuffer cb, uint32_t f, uint32_t imageIndex,
    vk::Semaphore imageAvail, vk::Semaphore renderDone,
    vk::Fence fence, bool first)
{
    SubmitAndPresent(m_device, m_swapchain, cb, f, imageIndex,
                     imageAvail, renderDone, fence, first,
                     m_computeQueueFamily, m_presentQueueFamily,
                     m_timestampPool, m_timestampPeriod, m_hasTimestamps,
                     m_frameCount, m_lastGpuMilliseconds);
}
