#include "render/ScreenshotCapture.h"
#include "core/GPUBuffer.h"
#include "core/Log.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstring>

ScreenshotCapture::ScreenshotCapture(
    const vk::raii::Device& d, const vk::raii::PhysicalDevice& p,
    uint32_t qf) : m_device(d), m_physDevice(p), m_queueFamily(qf) {}

ScreenshotCapture::TempImages ScreenshotCapture::createTempImages(
    uint32_t w, uint32_t h)
{
    auto create = [&](vk::Format fmt) {
        vk::ImageCreateInfo ci({}, vk::ImageType::e2D, fmt,
            {w, h, 1}, 1, 1, vk::SampleCountFlagBits::e1,
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
                vk::ImageViewType::e2D, fmt, vk::ComponentMapping{},
                vk::ImageSubresourceRange(
                    vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
        return std::make_tuple(
            std::move(img), std::move(mem), std::move(view));
    };
    TempImages t;
    std::tie(t.img, t.mem, t.view) = create(vk::Format::eR8G8B8A8Unorm);
    std::tie(t.nrm, t.nrmMem, t.nrmView) =
        create(vk::Format::eR16G16B16A16Sfloat);
    std::tie(t.dep, t.depMem, t.depView) = create(vk::Format::eR32Sfloat);
    return t;
}

void ScreenshotCapture::renderOneFrame(
    uint32_t f, uint32_t gx, uint32_t gy,
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

    if (f == 0) {
        for (auto* im : {&*temps.img, &*temps.nrm, &*temps.dep}) {
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

    traceAndDenoise(*cb, f, gx, gy, as, pipeline, camera, temps);

    cb.end();
    m_device.getQueue(m_queueFamily, 0)
        .submit(vk::SubmitInfo({}, {}, *cb), nullptr);
    m_device.getQueue(m_queueFamily, 0).waitIdle();
}

void ScreenshotCapture::traceAndDenoise(
    vk::CommandBuffer cb, uint32_t f, uint32_t gx, uint32_t gy,
    const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    const TempImages& temps)
{
    struct PC { float camOrigin[3]; float _pad0;
        float camU[3]; float _pad1; float camV[3]; float _pad2;
        float camW[3]; float _pad3;
        int spp, maxBounces, matCount;
        float fovTan, splitMult, forceSplitWidth;
        int scatterSamples; float mergeThreshold;
        int frameIndex;
        float diffuseStrength, specularStrength; int numLights; } pc{};
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
    pc.mergeThreshold=0.999f; pc.frameIndex=static_cast<int>(f);
    pc.diffuseStrength=as.getDiffuseStrength();
    pc.specularStrength=as.getSpecularStrength();
    pc.numLights=static_cast<int>(as.getLightCount());

    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.getPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.desc().pipelineLayout(), 0,
        pipeline.desc().descriptorSet(0), nullptr);
    cb.pushConstants<PC>(pipeline.desc().pipelineLayout(),
        vk::ShaderStageFlagBits::eCompute, 0, pc);
    cb.dispatch(gx, gy, 1);

    // Barrier: trace → denoise
    { auto sub = vk::ImageSubresourceRange(
          vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
      vk::ImageMemoryBarrier gb[3] = {
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *temps.nrm, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *temps.dep, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *temps.img, sub}};
      cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
          vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, gb); }

    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.getDenoisePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.desc().pipelineLayout(), 0,
        pipeline.desc().descriptorSet(0), nullptr);
    cb.dispatch(gx, gy, 1);

    // Barrier: denoise → transfer
    { vk::ImageMemoryBarrier bar(vk::AccessFlagBits::eShaderWrite,
          vk::AccessFlagBits::eTransferRead,
          vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          *temps.img, vk::ImageSubresourceRange(
              vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
      cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
          vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, bar); }
}

void ScreenshotCapture::readbackToPNG(
    const std::string& path, vk::Image img, uint32_t w, uint32_t h,
    const vk::raii::Queue& q)
{
    vk::DeviceSize ib = w * h * 4;
    auto stg = GPUBuffer::create(m_device, ib,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, m_physDevice);
    auto cp = vk::raii::CommandPool(m_device,
        {vk::CommandPoolCreateFlagBits::eTransient, m_queueFamily});
    auto cbs = vk::raii::CommandBuffers(m_device,
        {*cp, vk::CommandBufferLevel::ePrimary, 1});
    cbs[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cbs[0].copyImageToBuffer(img, vk::ImageLayout::eGeneral,
        *stg.buffer, vk::BufferImageCopy(0, 0, 0,
            vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor, 0, 0, 1),
            {0,0,0}, {w, h, 1}));
    cbs[0].end();
    q.submit(vk::SubmitInfo({}, {}, *cbs[0]), nullptr);
    q.waitIdle();

    void* m = stg.memory.mapMemory(0, ib);
    int r = stbi_write_png(path.c_str(), (int)w, (int)h,
                           4, m, (int)w * 4);
    stg.memory.unmapMemory();

    if (!r) Log::error("[Screenshot] Failed: {}", path);
    else Log::info("[Screenshot] Saved: {} ({}x{})", path, w, h);
}

void ScreenshotCapture::capture(
    const std::string& path, const AccelerationStructure& as,
    RayTracingPipeline& pipeline, const CameraParams& camera,
    uint32_t cw, uint32_t ch, uint32_t nFrames,
    vk::Image mainOut, vk::ImageView mainOutView,
    vk::Buffer mainAccBuf, vk::DeviceSize mainAccSz,
    vk::ImageView mainNrmView, vk::ImageView mainDepView)
{
    Log::info("[Screenshot] {}x{} x{}...", cw, ch, nFrames);
    auto temps = createTempImages(cw, ch);
    auto q = m_device.getQueue(m_queueFamily, 0);

    vk::DeviceSize asz = cw * ch * 4 * sizeof(float);
    auto capAcc = GPUBuffer::create(m_device, asz,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);
    auto capStg = GPUBuffer::create(m_device, asz,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, m_physDevice);
    { void* m = capStg.memory.mapMemory(0, asz);
      std::memset(m, 0, (size_t)asz); capStg.memory.unmapMemory();
      auto cp = vk::raii::CommandPool(m_device,
          {vk::CommandPoolCreateFlagBits::eTransient, m_queueFamily});
      auto cbs = vk::raii::CommandBuffers(m_device,
          {*cp, vk::CommandBufferLevel::ePrimary, 1});
      cbs[0].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      cbs[0].copyBuffer(*capStg.buffer, *capAcc.buffer,
                        vk::BufferCopy(0, 0, asz));
      cbs[0].end(); q.submit(vk::SubmitInfo({}, {}, *cbs[0]), nullptr);
      q.waitIdle(); }

    pipeline.desc().bindOutputImage(0, *temps.view, nullptr);
    pipeline.desc().bindAccumBuffer(0, *capAcc.buffer, asz);
    pipeline.desc().bindNormalImage(0, *temps.nrmView);
    pipeline.desc().bindDepthImage(0, *temps.depView);

    uint32_t gx = (cw+7)/8, gy = (ch+7)/8;
    for (uint32_t f = 0; f < nFrames; ++f)
        renderOneFrame(f, gx, gy, as, pipeline, camera, temps);

    readbackToPNG(path, *temps.img, cw, ch, q);

    pipeline.desc().bindOutputImage(0, mainOutView, nullptr);
    pipeline.desc().bindAccumBuffer(0, mainAccBuf, mainAccSz);
    pipeline.desc().bindNormalImage(0, mainNrmView);
    pipeline.desc().bindDepthImage(0, mainDepView);
    Log::info("[Screenshot] Done.");
}
