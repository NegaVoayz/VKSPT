#include "render/CaptureImageIO.h"
#include "core/GPUBuffer.h"
#include "core/Log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstring>

namespace {

void createOneCaptureImage(
    const vk::raii::Device& device, const vk::raii::PhysicalDevice& physDevice,
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
    vk::raii::Image img(device, ci);
    auto reqs = img.getMemoryRequirements();
    auto props = physDevice.getMemoryProperties();
    uint32_t mti = 0;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((reqs.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             vk::MemoryPropertyFlagBits::eDeviceLocal))
        { mti = i; break; }
    vk::raii::DeviceMemory mem(device,
        vk::MemoryAllocateInfo(reqs.size, mti));
    img.bindMemory(*mem, 0);
    vk::raii::ImageView view(device,
        vk::ImageViewCreateInfo({}, *img,
            vk::ImageViewType::e2D, format, vk::ComponentMapping{},
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
    outImage = std::move(img);
    outMemory = std::move(mem);
    outView = std::move(view);
}

} // namespace

CaptureTempImages CreateCaptureTempImages(
    const vk::raii::Device& device, const vk::raii::PhysicalDevice& physDevice,
    uint32_t width, uint32_t height)
{
    CaptureTempImages t;
    createOneCaptureImage(device, physDevice, width, height,
        vk::Format::eR8G8B8A8Unorm, t.image, t.memory, t.view);
    createOneCaptureImage(device, physDevice, width, height,
        vk::Format::eR16G16B16A16Sfloat, t.normal, t.normalMemory, t.normalView);
    createOneCaptureImage(device, physDevice, width, height,
        vk::Format::eR32Sfloat, t.depth, t.depthMemory, t.depthView);
    return t;
}

void TraceAndDenoiseCapture(
    vk::CommandBuffer cb, uint32_t frameIndex,
    uint32_t groupCountX, uint32_t groupCountY,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    const CameraParams& camera, const CaptureTempImages& temps)
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
    pc.gpuMilliseconds = 0.0f;

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

void ReadbackCapturePNG(
    const vk::raii::Device& device, const vk::raii::PhysicalDevice& physDevice,
    uint32_t queueFamily, const std::string& path,
    vk::Image img, uint32_t w, uint32_t h)
{
    vk::DeviceSize ib = w * h * 4;
    auto stg = GPUBuffer::Create(device, ib,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, physDevice);
    auto cp = vk::raii::CommandPool(device,
        {vk::CommandPoolCreateFlagBits::eTransient, queueFamily});
    auto cbs = vk::raii::CommandBuffers(device,
        {*cp, vk::CommandBufferLevel::ePrimary, 1});
    cbs[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cbs[0].copyImageToBuffer(img, vk::ImageLayout::eGeneral,
        *stg.Buffer, vk::BufferImageCopy(0, 0, 0,
            vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor, 0, 0, 1),
            {0,0,0}, {w, h, 1}));
    cbs[0].end();

    auto q = device.getQueue(queueFamily, 0);
    q.submit(vk::SubmitInfo({}, {}, *cbs[0]), nullptr);
    q.waitIdle();

    void* m = stg.Memory.mapMemory(0, ib);
    int r = stbi_write_png(path.c_str(), (int)w, (int)h,
                           4, m, (int)w * 4);
    stg.Memory.unmapMemory();

    if (!r) Log::error("[Screenshot] Failed: {}", path);
    else Log::info("[Screenshot] Saved: {} ({}x{})", path, w, h);
}
