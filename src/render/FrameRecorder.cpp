#include "render/FrameRecorder.h"
#include "core/Log.h"

static constexpr uint32_t TS_PER_FRAME = 2;

void FrameRecorder::init(
    const vk::raii::Device&         device,
    const vk::raii::SwapchainKHR&   swapchain,
    const std::vector<vk::Image>&   swapchainImages,
    vk::Image out, vk::Image nrm, vk::Image dep,
    uint32_t w, uint32_t h, uint32_t cqf, uint32_t pqf,
    const vk::raii::QueryPool* tp, float tsPeriod, bool hasTS)
{
    m_device = &device;  m_swapchain = &swapchain;
    m_swapImages = &swapchainImages;
    m_outImg = out;  m_nrmImg = nrm;  m_depImg = dep;
    m_w = w;  m_h = h;  m_cqf = cqf;  m_pqf = pqf;
    m_tsPool = tp;  m_tsPeriod = tsPeriod;  m_hasTS = hasTS;
}

void FrameRecorder::record(
    vk::CommandBuffer cb, uint32_t f, uint32_t imageIndex,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    const CameraParams& camera, int accumFrame, bool first)
{
    transitionImages(cb, first);
    dispatchTrace(cb, f, as, pipeline, camera, accumFrame);
    denoisePass(cb, f, pipeline);
    copyOutputToSwapchain(cb, imageIndex);
}

void FrameRecorder::transitionImages(vk::CommandBuffer cb, bool first)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    if (first) {
        for (auto img : {m_outImg, m_nrmImg, m_depImg}) {
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
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_outImg, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, b);
    }
}

void FrameRecorder::dispatchTrace(
    vk::CommandBuffer cb, uint32_t f,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    const CameraParams& camera, int accumFrame)
{
    struct PC {
        float camOrigin[3]; float _pad0;
        float camU[3];      float _pad1;
        float camV[3];      float _pad2;
        float camW[3];      float _pad3;
        int   spp, maxBounces, matCount;
        float fovTan, splitMult, forceSplitWidth;
        int   scatterSamples; float mergeThreshold;
        int   frameIndex; int _padEnd[2];
    } pc{};
    pc.camOrigin[0]=camera.origin[0]; pc.camOrigin[1]=camera.origin[1];
    pc.camOrigin[2]=camera.origin[2];
    pc.camU[0]=camera.camU[0]; pc.camU[1]=camera.camU[1];
    pc.camU[2]=camera.camU[2];
    pc.camV[0]=camera.camV[0]; pc.camV[1]=camera.camV[1];
    pc.camV[2]=camera.camV[2];
    pc.camW[0]=camera.camW[0]; pc.camW[1]=camera.camW[1];
    pc.camW[2]=camera.camW[2];
    pc.spp=4; pc.maxBounces=24;
    pc.matCount=static_cast<int>(as.getMaterialCount());
    pc.fovTan=0.57735f; pc.splitMult=1.0f;
    pc.forceSplitWidth=0.0f; pc.scatterSamples=1;
    pc.mergeThreshold=0.999f; pc.frameIndex=accumFrame;

    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.getPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.desc().pipelineLayout(), 0,
        pipeline.desc().descriptorSet(f), nullptr);
    cb.pushConstants<PC>(pipeline.desc().pipelineLayout(),
        vk::ShaderStageFlagBits::eCompute, 0, pc);

    if (m_hasTS) {
        cb.resetQueryPool(*m_tsPool, f * TS_PER_FRAME, TS_PER_FRAME);
        cb.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                          *m_tsPool, f * TS_PER_FRAME);
    }
    uint32_t gx = (m_w + 7) / 8, gy = (m_h + 7) / 8;
    cb.dispatch(gx, gy, 1);
}

void FrameRecorder::denoisePass(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageMemoryBarrier gb[3] = {
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_nrmImg, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_depImg, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_outImg, sub},
    };
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, gb);

    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.getDenoisePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.desc().pipelineLayout(), 0,
        pipeline.desc().descriptorSet(f), nullptr);
    uint32_t gx = (m_w + 7) / 8, gy = (m_h + 7) / 8;
    cb.dispatch(gx, gy, 1);

    if (m_hasTS)
        cb.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                          *m_tsPool, f * TS_PER_FRAME + 1);
}

void FrameRecorder::copyOutputToSwapchain(
    vk::CommandBuffer cb, uint32_t imageIndex)
{
    auto sub = vk::ImageSubresourceRange(
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

    // Denoise → transfer
    {
        vk::ImageMemoryBarrier ob(vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_outImg, sub);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, ob);
    }

    vk::Image sw = (*m_swapImages)[imageIndex];

    // Undefined → TransferDst
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
        {0,0,0}, {m_w, m_h, 1});
    cb.copyImage(m_outImg, vk::ImageLayout::eGeneral,
                 sw, vk::ImageLayout::eTransferDstOptimal, region);

    // TransferDst → PresentSrc
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
    vk::PipelineStageFlags ws = vk::PipelineStageFlagBits::eComputeShader;
    m_device->getQueue(m_cqf, 0)
        .submit(vk::SubmitInfo(imageAvail, ws, cb, renderDone), fence);
    m_device->getQueue(m_pqf, 0)
        .presentKHR(vk::PresentInfoKHR(renderDone, **m_swapchain, imageIndex));

    if (first)
        m_device->waitForFences(fence, true, UINT64_MAX);

    if (m_hasTS && !first) {
        uint64_t r[2];
        auto res = static_cast<vk::Device>(**m_device).getQueryPoolResults(
            *m_tsPool, f * TS_PER_FRAME, TS_PER_FRAME,
            sizeof(r), r, sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (res == vk::Result::eSuccess && r[0] > 0 && r[1] > r[0]) {
            float ms = float(r[1] - r[0]) * m_tsPeriod / 1e6f;
            m_frameCount++;
            if (m_frameCount % 60 == 0)
                Log::info("[GPU] frame {} compute time: {} ms",
                          m_frameCount, ms);
        }
    }
}
