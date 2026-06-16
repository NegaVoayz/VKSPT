#include "render/ScreenshotCapture.h"
#include "core/GPUBuffer.h"
#include "core/Log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstring>

ScreenshotCapture::ScreenshotCapture(
    const vk::raii::Device& d, const vk::raii::PhysicalDevice& p,
    uint32_t qf) : m_device(d), m_physDevice(p), m_queueFamily(qf) {}

void ScreenshotCapture::createOneTempImage(
    uint32_t width, uint32_t height, vk::Format format,
    vk::raii::Image& outImage,
    vk::raii::DeviceMemory& outMemory,
    vk::raii::ImageView& outView)
{
    vk::ImageCreateInfo ci({}, vk::ImageType::e2D, format,
        {width, height, 1}, 1, 1, vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive);
    vk::raii::Image img(m_device, ci);
    auto reqs = img.getMemoryRequirements();
    auto props = m_physDevice.getMemoryProperties();
    uint32_t mti = 0;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((reqs.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             vk::MemoryPropertyFlagBits::eDeviceLocal))
        { mti = i; break; }
    vk::raii::DeviceMemory mem(m_device,
        vk::MemoryAllocateInfo(reqs.size, mti));
    img.bindMemory(*mem, 0);
    vk::raii::ImageView view(m_device,
        vk::ImageViewCreateInfo({}, *img,
            vk::ImageViewType::e2D, format, vk::ComponentMapping{},
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
    outImage = std::move(img);
    outMemory = std::move(mem);
    outView = std::move(view);
}

ScreenshotCapture::TempImages ScreenshotCapture::createTempImages(
    uint32_t width, uint32_t height)
{
    TempImages t;
    createOneTempImage(width, height, vk::Format::eR8G8B8A8Unorm,
                       t.image, t.memory, t.view);
    createOneTempImage(width, height, vk::Format::eR16G16B16A16Sfloat,
                       t.normal, t.normalMemory, t.normalView);
    createOneTempImage(width, height, vk::Format::eR32Sfloat,
                       t.depth, t.depthMemory, t.depthView);
    return t;
}

void ScreenshotCapture::renderOneFrame(
    uint32_t frameIndex, uint32_t groupCountX, uint32_t groupCountY,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    const TempImages& temps)
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

    traceAndDenoise(*cb, frameIndex, groupCountX, groupCountY,
                    as, pipeline, camera, temps);

    cb.end();
    m_device.getQueue(m_queueFamily, 0)
        .submit(vk::SubmitInfo({}, {}, *cb), nullptr);
    m_device.getQueue(m_queueFamily, 0).waitIdle();
}

void ScreenshotCapture::traceAndDenoise(
    vk::CommandBuffer cb, uint32_t frameIndex,
    uint32_t groupCountX, uint32_t groupCountY,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    const TempImages& temps)
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
    pc.mergeThreshold=0.999f; pc.frameIndex=static_cast<int>(frameIndex);
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
    pc.fps = 0.0f;

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(0), nullptr);
    cb.pushConstants<FrameRecorder::TracePushConstant>(
        pipeline.Desc().PipelineLayout(),
        vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eClosestHitKHR |
            vk::ShaderStageFlagBits::eMissKHR |
            vk::ShaderStageFlagBits::eCompute, 0, pc);

    uint32_t w = groupCountX * 8, h = groupCountY * 8;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdTraceRaysKHR(
        static_cast<VkCommandBuffer>(cb),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.RaygenRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.MissRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.HitRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.CallableRegion()),
        w, h, 1);

    { auto sub = vk::ImageSubresourceRange(
          vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
      vk::ImageMemoryBarrier gb[3] = {
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *temps.normal, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *temps.depth, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *temps.image, sub}};
      cb.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
          vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, gb); }

    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetDenoisePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(0), nullptr);
    cb.dispatch(groupCountX, groupCountY, 1);

    { vk::ImageMemoryBarrier bar(vk::AccessFlagBits::eShaderWrite,
          vk::AccessFlagBits::eTransferRead,
          vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          *temps.image, vk::ImageSubresourceRange(
              vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
      cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
          vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, bar); }
}

void ScreenshotCapture::readbackToPNG(
    const std::string& path, vk::Image img, uint32_t w, uint32_t h,
    const vk::raii::Queue& q)
{
    vk::DeviceSize ib = w * h * 4;
    auto stg = GPUBuffer::Create(m_device, ib,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, m_physDevice);
    auto cp = vk::raii::CommandPool(m_device,
        {vk::CommandPoolCreateFlagBits::eTransient, m_queueFamily});
    auto cbs = vk::raii::CommandBuffers(m_device,
        {*cp, vk::CommandBufferLevel::ePrimary, 1});
    cbs[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cbs[0].copyImageToBuffer(img, vk::ImageLayout::eGeneral,
        *stg.Buffer, vk::BufferImageCopy(0, 0, 0,
            vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor, 0, 0, 1),
            {0,0,0}, {w, h, 1}));
    cbs[0].end();
    q.submit(vk::SubmitInfo({}, {}, *cbs[0]), nullptr);
    q.waitIdle();

    void* m = stg.Memory.mapMemory(0, ib);
    int r = stbi_write_png(path.c_str(), (int)w, (int)h,
                           4, m, (int)w * 4);
    stg.Memory.unmapMemory();

    if (!r) Log::error("[Screenshot] Failed: {}", path);
    else Log::info("[Screenshot] Saved: {} ({}x{})", path, w, h);
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
    auto temps = createTempImages(targetWidth, targetHeight);
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

    readbackToPNG(path, *temps.image, targetWidth, targetHeight, q);

    // Restore main-resolution bindings
    pipeline.Desc().BindOutputImage(0, mainOutputView, nullptr);
    pipeline.Desc().BindAccumBuffer(0, mainAccumBuffer, mainAccumSize);
    pipeline.Desc().BindNormalImage(0, mainNormalView);
    pipeline.Desc().BindDepthImage(0, mainDepthView);
    // Photon buffers stay bound (same buffers, same set)
    Log::info("[Screenshot] Done.");
}
