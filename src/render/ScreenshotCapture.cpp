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

    traceAndDenoise(*cb, frameIndex, groupCountX, groupCountY,
                    as, pipeline, camera, temps);

    cb.end();
    m_device.getQueue(m_queueFamily, 0)
        .submit(vk::SubmitInfo({}, {}, *cb), nullptr);
    m_device.getQueue(m_queueFamily, 0).waitIdle();
}

namespace { struct TracePC {
    float camOrigin[3]; float pad0;
    float camU[3]; float pad1; float camV[3]; float pad2;
    float camW[3]; float pad3;
    int spp, maxBounces, matCount;
    float fovTan, splitMult, forceSplitWidth;
    int scatterSamples; float mergeThreshold;
    int frameIndex;
    float diffuseStrength, specularStrength; int numLights;
    float minSplitNm;
}; }

static void buildPushConstants(
    const AccelerationStructure& as, const CameraParams& camera,
    uint32_t frameIndex, TracePC& pc)
{
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
    pc.forceSplitWidth=20.0f; pc.scatterSamples=1;
    pc.mergeThreshold=0.999f; pc.frameIndex=static_cast<int>(frameIndex);
    pc.diffuseStrength=as.getDiffuseStrength();
    pc.specularStrength=as.getSpecularStrength();
    pc.numLights=static_cast<int>(as.getLightCount());
    pc.minSplitNm=20.0f;
    pc.forceSplitWidth=10.0f;
}

void ScreenshotCapture::traceAndDenoise(
    vk::CommandBuffer cb, uint32_t frameIndex,
    uint32_t groupCountX, uint32_t groupCountY,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    const TempImages& temps)
{
    TracePC pc{};
    buildPushConstants(as, camera, frameIndex, pc);

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(0), nullptr);
    cb.pushConstants<TracePC>(pipeline.Desc().PipelineLayout(),
        vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eClosestHitKHR |
            vk::ShaderStageFlagBits::eMissKHR |
            vk::ShaderStageFlagBits::eCompute, 0, pc);

    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdTraceRaysKHR(
        static_cast<VkCommandBuffer>(cb),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.RaygenRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.MissRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.HitRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.CallableRegion()),
        groupCountX * 8, groupCountY * 8, 1);

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
      cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
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

    pipeline.Desc().BindOutputImage(0, *temps.view, nullptr);
    pipeline.Desc().BindAccumBuffer(0, *capAcc.Buffer, asz);
    pipeline.Desc().BindNormalImage(0, *temps.normalView);
    pipeline.Desc().BindDepthImage(0, *temps.depthView);

    uint32_t gx = (targetWidth+7)/8, gy = (targetHeight+7)/8;
    for (uint32_t f = 0; f < frameCount; ++f)
        renderOneFrame(f, gx, gy, as, pipeline, camera, temps);

    readbackToPNG(path, *temps.image, targetWidth, targetHeight, q);

    pipeline.Desc().BindOutputImage(0, mainOutputView, nullptr);
    pipeline.Desc().BindAccumBuffer(0, mainAccumBuffer, mainAccumSize);
    pipeline.Desc().BindNormalImage(0, mainNormalView);
    pipeline.Desc().BindDepthImage(0, mainDepthView);
    Log::info("[Screenshot] Done.");
}
