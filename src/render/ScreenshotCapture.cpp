#include "render/ScreenshotCapture.h"
#include "core/GPUBuffer.h"
#include "core/Log.h"

ScreenshotCapture::ScreenshotCapture(
    const vk::raii::Device& d, const vk::raii::PhysicalDevice& p,
    uint32_t qf) : m_device(d), m_physDevice(p), m_queueFamily(qf) {}

void ScreenshotCapture::renderOneFrame(
    uint32_t frameIndex, uint32_t groupCountX, uint32_t groupCountY,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    const CaptureTempImages& temps)
{
    auto cp = vk::raii::CommandPool(m_device,
        {vk::CommandPoolCreateFlagBits::eTransient |
         vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         m_queueFamily});
    auto cbs = vk::raii::CommandBuffers(m_device,
        {*cp, vk::CommandBufferLevel::ePrimary, 1});
    auto& cb = cbs[0];
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    if (frameIndex == 0) {
        for (auto* im : {&*temps.image, &*temps.normal, &*temps.depth}) {
            vk::ImageMemoryBarrier b({},
                vk::AccessFlagBits::eShaderWrite,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                *im, vk::ImageSubresourceRange(
                    vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eComputeShader,
                {}, {}, {}, b);
        }
    }

    // blendPhoton: EMA photon display data
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetBlendPhotonPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(0), nullptr);
    cb.dispatch(2048, 1, 1);

    {
        vk::BufferMemoryBarrier b(vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getDisplayCellData().Buffer, 0, as.getDisplayCellData().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            {}, {}, b, {});
    }

    TraceAndDenoiseCapture(*cb, frameIndex, groupCountX, groupCountY,
                           as, pipeline, camera, temps);

    cb.end();
    m_device.getQueue(m_queueFamily, 0)
        .submit(vk::SubmitInfo({}, {}, *cb), nullptr);
    m_device.getQueue(m_queueFamily, 0).waitIdle();
}

void ScreenshotCapture::capture(
    const std::string& path, const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    uint32_t targetWidth, uint32_t targetHeight, uint32_t frameCount,
    vk::Image mainOutput, vk::ImageView mainOutputView,
    vk::Buffer mainAccumBuffer, vk::DeviceSize mainAccumSize,
    vk::ImageView mainNormalView, vk::ImageView mainDepthView)
{
    Log::info("[Screenshot] {}x{} x{}...", targetWidth, targetHeight, frameCount);
    auto temps = CreateCaptureTempImages(m_device, m_physDevice,
                                         targetWidth, targetHeight);
    auto q = m_device.getQueue(m_queueFamily, 0);

    vk::DeviceSize asz = targetWidth * targetHeight * 4 * sizeof(float);
    auto capAcc = GPUBuffer::Create(m_device, asz,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);
    auto capStg = GPUBuffer::Create(m_device, asz,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, m_physDevice);
    { void* m = capStg.Memory.mapMemory(0, asz);
      std::memset(m, 0, (size_t)asz); capStg.Memory.unmapMemory();
      auto cp = vk::raii::CommandPool(m_device,
          {vk::CommandPoolCreateFlagBits::eTransient, m_queueFamily});
      auto cbs = vk::raii::CommandBuffers(m_device,
          {*cp, vk::CommandBufferLevel::ePrimary, 1});
      cbs[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      cbs[0].copyBuffer(*capStg.Buffer, *capAcc.Buffer,
                        vk::BufferCopy(0, 0, asz));
      cbs[0].end(); q.submit(vk::SubmitInfo({}, {}, *cbs[0]), nullptr);
      q.waitIdle(); }

    // Bind all resources into set 0: output images + photon buffers
    pipeline.Desc().BindOutputImage(0, *temps.view, nullptr);
    pipeline.Desc().BindAccumBuffer(0, *capAcc.Buffer, asz);
    pipeline.Desc().BindNormalImage(0, *temps.normalView);
    pipeline.Desc().BindDepthImage(0, *temps.depthView);
    pipeline.Desc().BindPhotonBuffer(
        0, *as.getPhotonBuffer().Buffer, as.getPhotonBuffer().Size);
    pipeline.Desc().BindPhotonCounter(
        0, *as.getPhotonCounter().Buffer, as.getPhotonCounter().Size);
    pipeline.Desc().BindHashCellData(
        0, *as.getHashCellData().Buffer, as.getHashCellData().Size);
    pipeline.Desc().BindSortedPhotonIndices(
        0, *as.getSortedPhotonIndices().Buffer, as.getSortedPhotonIndices().Size);
    pipeline.Desc().BindCellPhotonData(
        0, *as.getCellPhotonData().Buffer, as.getCellPhotonData().Size);
    pipeline.Desc().BindGatheredCellData(
        0, *as.getGatheredCellData().Buffer, as.getGatheredCellData().Size);
    pipeline.Desc().BindDisplayCellData(
        0, *as.getDisplayCellData().Buffer, as.getDisplayCellData().Size);
    pipeline.Desc().BindRayStats(
        0, *as.getRayStats().Buffer, as.getRayStats().Size);

    uint32_t gx = (targetWidth+7)/8, gy = (targetHeight+7)/8;
    for (uint32_t f = 0; f < frameCount; ++f)
        renderOneFrame(f, gx, gy, as, pipeline, camera, temps);

    ReadbackCapturePNG(m_device, m_physDevice, m_queueFamily,
                       path, *temps.image, targetWidth, targetHeight);

    // Restore main-resolution bindings
    pipeline.Desc().BindOutputImage(0, mainOutputView, nullptr);
    pipeline.Desc().BindAccumBuffer(0, mainAccumBuffer, mainAccumSize);
    pipeline.Desc().BindNormalImage(0, mainNormalView);
    pipeline.Desc().BindDepthImage(0, mainDepthView);
    // Photon buffers stay bound (same buffers, same set)
    Log::info("[Screenshot] Done.");
}
