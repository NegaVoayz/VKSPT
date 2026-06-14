#include "render/FrameRecorder.h"
#include "ray/DescriptorManager.h"
#include "core/Log.h"

static constexpr uint32_t TS_PER_FRAME = 2;

FrameRecorder::FrameRecorder(
    const vk::raii::Device&         device,
    const vk::raii::SwapchainKHR&   swapchain,
    const std::vector<vk::Image>&   swapchainImages,
    vk::Image out, vk::Image nrm, vk::Image dep,
    uint32_t w, uint32_t h, uint32_t cqf, uint32_t pqf,
    const vk::raii::QueryPool* tp, float tsPeriod, bool hasTS)
    : m_device(device)
    , m_swapchain(swapchain)
    , m_swapImages(swapchainImages)
    , m_outImg(out)
    , m_nrmImg(nrm)
    , m_depImg(dep)
    , m_w(w)
    , m_h(h)
    , m_cqf(cqf)
    , m_pqf(pqf)
    , m_tsPool(tp)
    , m_tsPeriod(tsPeriod)
    , m_hasTS(hasTS)
    , m_photonFence(m_device, vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled})
{}

void FrameRecorder::record(
    vk::CommandBuffer cb, uint32_t f, uint32_t imageIndex,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    const CameraParams& camera, int accumFrame, bool first, bool showStats, float fps)
{
    transitionImages(cb, first);

    // Reset ray stats
    cb.fillBuffer(*as.getRayStats().Buffer, 0, as.getRayStats().Size, 0);

    // Zero gatheredCellData on first frame (subsequent batches accumulate via atomic CAS)
    if (first)
        cb.fillBuffer(*as.getGatheredCellData().Buffer, 0,
                       as.getGatheredCellData().Size, 0);

    // Barrier: transfer → RT
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
        }
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR |
                vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, bArr, {});
    }

    // Camera trace (O(1) cell-lookup into gatheredCellData)
    dispatchTrace(cb, f, as, pipeline, camera, accumFrame, fps);
    denoisePass(cb, f, pipeline);

    // Stats overlay (F3 toggle)
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

void FrameRecorder::setupPhotons(
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    vk::ImageView outView, vk::ImageView nrmView, vk::ImageView depView,
    vk::Buffer accumBuf, vk::DeviceSize accumSz)
{
    m_photonOutView = outView;
    m_photonNrmView = nrmView;
    m_photonDepView = depView;
    m_photonAccumBuf = accumBuf;
    m_photonAccumSz  = accumSz;

    // Compute active light count and per-light photons
    const auto& cpuLights = as.getLightsCPU();
    int totalLights = static_cast<int>(as.getLightCount());
    m_activeLightCount = 0;
    for (int i = 0; i < totalLights; i++)
        if (static_cast<int>(cpuLights[i].pos_type[3]) != 3 &&
            cpuLights[i].color_intensity[3] > 0.0f) m_activeLightCount++;
    if (m_activeLightCount == 0) m_activeLightCount = 1;
    m_perLight = m_photonCount / m_activeLightCount;

    m_nextLightIndex = 0;

    // Create photon command pool, command buffer, and fence
    m_photonCmdPool = vk::raii::CommandPool(m_device,
        {vk::CommandPoolCreateFlagBits::eTransient |
         vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         m_cqf});
    auto cbs = vk::raii::CommandBuffers(m_device,
        {*m_photonCmdPool, vk::CommandBufferLevel::ePrimary, 1});
    m_photonCmdBuf = std::move(cbs[0]);

    m_photonDone = false;

    // Bind photon descriptor set with all required resources
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    pipeline.Desc().BindTLAS(ps, as.getTLAS());
    pipeline.Desc().BindMaterialBuffer(ps, *as.getMaterialBuffer().Buffer,
                                        as.getMaterialBuffer().Size);
    pipeline.Desc().BindLightBuffer(ps, *as.getLightBuffer().Buffer,
                                     as.getLightBuffer().Size);
    pipeline.Desc().BindGeometrySSBOs(
        ps, *as.getGeometry().vertexBuf().Buffer,
            as.getGeometry().vertexBuf().Size,
            *as.getGeometry().indexBuf().Buffer,
            as.getGeometry().indexBuf().Size,
            *as.getGeometry().rangeBuf().Buffer,
            as.getGeometry().rangeBuf().Size);
    pipeline.Desc().BindEnvMap(ps, as.getEnvMap().view(),
                                as.getEnvMap().sampler());
    pipeline.Desc().BindNormalSSBO(ps, *as.getGeometry().normalBuf().Buffer,
                                    as.getGeometry().normalBuf().Size);
    pipeline.Desc().BindInstanceNormalBuffer(
        ps, *as.getInstanceNormalBuffer().Buffer,
            as.getInstanceNormalBuffer().Size);
    pipeline.Desc().BindPhotonBuffer(
        ps, *as.getPhotonBuffer().Buffer, as.getPhotonBuffer().Size);
    pipeline.Desc().BindPhotonCounter(
        ps, *as.getPhotonCounter().Buffer, as.getPhotonCounter().Size);
    pipeline.Desc().BindHashCellData(
        ps, *as.getHashCellData().Buffer, as.getHashCellData().Size);
    pipeline.Desc().BindSortedPhotonIndices(
        ps, *as.getSortedPhotonIndices().Buffer, as.getSortedPhotonIndices().Size);
    pipeline.Desc().BindCellPhotonData(
        ps, *as.getCellPhotonData().Buffer, as.getCellPhotonData().Size);
    pipeline.Desc().BindGatheredCellData(
        ps, *as.getGatheredCellData().Buffer, as.getGatheredCellData().Size);
    pipeline.Desc().BindRayStats(
        ps, *as.getRayStats().Buffer, as.getRayStats().Size);
    pipeline.Desc().BindOutputImage(ps, m_photonOutView, nullptr);
    pipeline.Desc().BindAccumBuffer(ps, m_photonAccumBuf, m_photonAccumSz);
    pipeline.Desc().BindNormalImage(ps, m_photonNrmView);
    pipeline.Desc().BindDepthImage(ps, m_photonDepView);
}

void FrameRecorder::trySubmitPhotonBatch(
    const AccelerationStructure& as, RayTracingPipeline& pipeline)
{
    if (m_photonDone)
        return;

    // Previous batch still in flight
    if (m_device.waitForFences(*m_photonFence, false, 0) != vk::Result::eSuccess)
        return;

    m_device.resetFences(*m_photonFence);

    // Find next active light
    const auto& cpuLights = as.getLightsCPU();
    int totalLights = static_cast<int>(as.getLightCount());
    int batchIdx = -1;
    for (int i = m_nextLightIndex; i < totalLights; i++) {
        if (static_cast<int>(cpuLights[i].pos_type[3]) != 3 &&
            cpuLights[i].color_intensity[3] > 0.0f) {
            batchIdx = i;
            break;
        }
    }

    if (batchIdx < 0) {
        m_photonDone = true;
        return;
    }

    // Update photon descriptor set (buffer addresses may change after swapchain resize etc.)
    constexpr auto ps = DescriptorManager::PHOTON_SET;
    pipeline.Desc().BindPhotonBuffer(
        ps, *as.getPhotonBuffer().Buffer, as.getPhotonBuffer().Size);
    pipeline.Desc().BindPhotonCounter(
        ps, *as.getPhotonCounter().Buffer, as.getPhotonCounter().Size);
    pipeline.Desc().BindHashCellData(
        ps, *as.getHashCellData().Buffer, as.getHashCellData().Size);
    pipeline.Desc().BindSortedPhotonIndices(
        ps, *as.getSortedPhotonIndices().Buffer, as.getSortedPhotonIndices().Size);
    pipeline.Desc().BindCellPhotonData(
        ps, *as.getCellPhotonData().Buffer, as.getCellPhotonData().Size);
    pipeline.Desc().BindGatheredCellData(
        ps, *as.getGatheredCellData().Buffer, as.getGatheredCellData().Size);
    pipeline.Desc().BindRayStats(
        ps, *as.getRayStats().Buffer, as.getRayStats().Size);

    // Record + submit one batch
    m_photonCmdBuf.reset({});
    recordPhotonBatch(*m_photonCmdBuf, as, pipeline,
                      batchIdx, m_perLight, m_activeLightCount);

    m_device.getQueue(m_cqf, 0)
        .submit(vk::SubmitInfo({}, {}, *m_photonCmdBuf), *m_photonFence);

    m_nextLightIndex = batchIdx + 1;
}

void FrameRecorder::recordPhotonBatch(
    vk::CommandBuffer cb,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    int lightIndex, int photonsPerBatch, int totalBatches)
{
    constexpr auto ps = DescriptorManager::PHOTON_SET;

    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Reset per-batch scratch buffers
    cb.fillBuffer(*as.getPhotonCounter().Buffer, 0, 4, 0);
    cb.fillBuffer(*as.getHashCellData().Buffer, 0, as.getHashCellData().Size, 0);

    // Barrier: transfer → RT|compute
    {
        std::vector<vk::BufferMemoryBarrier> bArr = {
            {vk::AccessFlagBits::eTransferWrite,
             vk::AccessFlagBits::eShaderWrite,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getPhotonCounter().Buffer, 0, 4},
            {vk::AccessFlagBits::eTransferWrite,
             vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getHashCellData().Buffer, 0, as.getHashCellData().Size},
        };
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR |
                vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, bArr, {});
    }

    // Trace photons for this batch
    dispatchPhotonTraceBatch(cb, ps, as, pipeline,
                             lightIndex, photonsPerBatch, totalBatches);

    // Barrier: RT → compute (photon buffer + counter → hash build)
    {
        std::vector<vk::BufferMemoryBarrier> bArr = {
            {vk::AccessFlagBits::eShaderWrite,
             vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getPhotonBuffer().Buffer, 0, as.getPhotonBuffer().Size},
            {vk::AccessFlagBits::eShaderWrite,
             vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getPhotonCounter().Buffer, 0, 4},
        };
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, bArr, {});
    }

    // Hash count
    dispatchHashCount(cb, ps, pipeline, photonsPerBatch);

    // Barrier: compute → compute (hashCellData → scan)
    {
        vk::BufferMemoryBarrier b(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getHashCellData().Buffer, 0, as.getHashCellData().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, b, {});
    }

    // Hash scan
    dispatchHashScan(cb, ps, pipeline);

    // Barrier: compute → compute (hashCellData offsets → scatter)
    {
        vk::BufferMemoryBarrier b(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getHashCellData().Buffer, 0, as.getHashCellData().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, b, {});
    }

    // Hash scatter
    dispatchHashScatter(cb, ps, pipeline, photonsPerBatch);

    // Barrier: compute → compute (hash data + sorted indices → aggregate)
    {
        std::vector<vk::BufferMemoryBarrier> bArr = {
            {vk::AccessFlagBits::eShaderWrite,
             vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getHashCellData().Buffer, 0, as.getHashCellData().Size},
            {vk::AccessFlagBits::eShaderWrite,
             vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getSortedPhotonIndices().Buffer, 0, as.getSortedPhotonIndices().Size},
        };
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, bArr, {});
    }

    // Hash aggregate
    dispatchHashAggregate(cb, ps, pipeline);

    // Barrier: compute → compute (cellPhotonData → hash_gather)
    {
        vk::BufferMemoryBarrier b(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getCellPhotonData().Buffer, 0, as.getCellPhotonData().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, b, {});
    }

    // Hash gather (atomic-CAS accumulate into gatheredCellData)
    dispatchHashGather(cb, ps, pipeline);

    // Barrier: compute → RT (gatheredCellData visible to subsequent camera trace)
    {
        vk::BufferMemoryBarrier b(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *as.getGatheredCellData().Buffer, 0, as.getGatheredCellData().Size);
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            {}, {}, b, {});
    }

    cb.end();
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
    const CameraParams& camera, int accumFrame, float fps)
{
    struct PC {
        float camOrigin[3]; float _pad0;
        float camU[3];      float _pad1;
        float camV[3];      float _pad2;
        float camW[3];      float _pad3;
        int   spp, maxBounces, matCount;
        float fovTan, splitMult, forceSplitWidth;
        int   scatterSamples; float mergeThreshold;
        int   frameIndex;
        float diffuseStrength, specularStrength; int numLights;
        float minSplitNm;
        int   passType;
        float minGatherRadius;
        float maxGatherRadius;
        float hashCellSize;
        int   photonMaxBounces;
        int   photonCount;
        float fps;
        int   batchIndex;
        int   batchCount;
        float lightIntensityPC;
        float lightColorPC[3];
        float _padEnd[22];
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
    pc.forceSplitWidth=0.05f; pc.scatterSamples=1;
    pc.mergeThreshold=0.999f; pc.frameIndex=accumFrame;
    pc.diffuseStrength=as.getDiffuseStrength();
    pc.specularStrength=as.getSpecularStrength();
    pc.numLights=static_cast<int>(as.getLightCount());
    pc.minSplitNm=20.0f;
    pc.passType = 0;  // PASS_CAMERA
    pc.minGatherRadius = m_minGatherRadius;
    pc.maxGatherRadius = m_maxGatherRadius;
    pc.hashCellSize = m_hashCellSize;
    pc.photonCount = m_photonCount;
    pc.photonMaxBounces = m_photonMaxBounces;
    pc.batchIndex = 0;
    pc.batchCount = 1;
    pc.lightIntensityPC = 0;
    pc.lightColorPC[0] = 0; pc.lightColorPC[1] = 0; pc.lightColorPC[2] = 0;
    pc.fps = fps;

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.pushConstants<PC>(pipeline.Desc().PipelineLayout(),
        vk::ShaderStageFlagBits::eRaygenKHR |
            vk::ShaderStageFlagBits::eClosestHitKHR |
            vk::ShaderStageFlagBits::eMissKHR |
            vk::ShaderStageFlagBits::eCompute, 0, pc);

    if (m_hasTS) {
        cb.resetQueryPool(*m_tsPool, f * TS_PER_FRAME, TS_PER_FRAME);
        cb.writeTimestamp(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
                          *m_tsPool, f * TS_PER_FRAME);
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdTraceRaysKHR(
        static_cast<VkCommandBuffer>(cb),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.RaygenRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.MissRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.HitRegion()),
        reinterpret_cast<const VkStridedDeviceAddressRegionKHR*>(&pipeline.CallableRegion()),
        m_w, m_h, 1);
}

void FrameRecorder::dispatchPhotonTrace(
    vk::CommandBuffer cb, uint32_t f,
    const AccelerationStructure& as, RayTracingPipeline& pipeline)
{
    struct PC {
        float camOrigin[3]; float _pad0;
        float camU[3];      float _pad1;
        float camV[3];      float _pad2;
        float camW[3];      float _pad3;
        int   spp, maxBounces, matCount;
        float fovTan, splitMult, forceSplitWidth;
        int   scatterSamples; float mergeThreshold;
        int   frameIndex;
        float diffuseStrength, specularStrength; int numLights;
        float minSplitNm;
        int   passType;
        float minGatherRadius;
        float maxGatherRadius;
        float hashCellSize;
        int   photonMaxBounces;
        int   photonCount;
        float fps;
        int   batchIndex;
        int   batchCount;
        float lightIntensityPC;
        float lightColorPC[3];
        float _padEnd[22];
    } pc{};

    pc.spp = 1; pc.maxBounces = m_photonMaxBounces;
    pc.matCount = static_cast<int>(as.getMaterialCount());
    pc.fovTan = 0.57735f; pc.splitMult = 1.0f;
    pc.forceSplitWidth = 0.005f; pc.scatterSamples = 1;
    pc.mergeThreshold = 0.999f; pc.frameIndex = 0;
    pc.diffuseStrength = as.getDiffuseStrength();
    pc.specularStrength = as.getSpecularStrength();
    pc.numLights = static_cast<int>(as.getLightCount());
    pc.minSplitNm = 20.0f;
    pc.passType = 1;  // PASS_PHOTON
    pc.minGatherRadius = m_minGatherRadius;
    pc.maxGatherRadius = m_maxGatherRadius;
    pc.hashCellSize = m_hashCellSize;
    pc.photonMaxBounces = m_photonMaxBounces;
    pc.photonCount = m_photonCount;
    pc.batchIndex = 0;
    pc.batchCount = 1;
    pc.lightIntensityPC = 0;
    pc.lightColorPC[0] = 0; pc.lightColorPC[1] = 0; pc.lightColorPC[2] = 0;

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.pushConstants<PC>(pipeline.Desc().PipelineLayout(),
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
        m_photonCount, 1, 1);
}

void FrameRecorder::dispatchPhotonTraceBatch(
    vk::CommandBuffer cb, uint32_t f,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    int batchIndex, int photonsPerBatch, int totalBatches)
{
    struct PC {
        float camOrigin[3]; float _pad0;
        float camU[3];      float _pad1;
        float camV[3];      float _pad2;
        float camW[3];      float _pad3;
        int   spp, maxBounces, matCount;
        float fovTan, splitMult, forceSplitWidth;
        int   scatterSamples; float mergeThreshold;
        int   frameIndex;
        float diffuseStrength, specularStrength; int numLights;
        float minSplitNm;
        int   passType;
        float minGatherRadius;
        float maxGatherRadius;
        float hashCellSize;
        int   photonMaxBounces;
        int   photonCount;
        float fps;
        int   batchIndex;
        int   batchCount;
        float lightIntensityPC;
        float lightColorPC[3];
        float _padEnd[22];
    } pc{};

    pc.spp = 1; pc.maxBounces = m_photonMaxBounces;
    pc.matCount = static_cast<int>(as.getMaterialCount());
    pc.fovTan = 0.57735f; pc.splitMult = 1.0f;
    pc.forceSplitWidth = 0.005f; pc.scatterSamples = 1;
    pc.mergeThreshold = 0.999f; pc.frameIndex = 0;
    pc.diffuseStrength = as.getDiffuseStrength();
    pc.specularStrength = as.getSpecularStrength();
    pc.numLights = static_cast<int>(as.getLightCount());
    pc.minSplitNm = 20.0f;
    pc.passType = 1;  // PASS_PHOTON
    pc.minGatherRadius = m_minGatherRadius;
    pc.maxGatherRadius = m_maxGatherRadius;
    pc.hashCellSize = m_hashCellSize;
    pc.photonMaxBounces = m_photonMaxBounces;
    pc.photonCount = photonsPerBatch;
    pc.batchIndex = batchIndex;
    pc.batchCount = totalBatches;
    auto& lcpu = as.getLightsCPU()[batchIndex];
    pc.lightIntensityPC = lcpu.color_intensity[3];
    pc.lightColorPC[0] = lcpu.color_intensity[0];
    pc.lightColorPC[1] = lcpu.color_intensity[1];
    pc.lightColorPC[2] = lcpu.color_intensity[2];

    cb.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR,
                    pipeline.GetRTPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.pushConstants<PC>(pipeline.Desc().PipelineLayout(),
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

void FrameRecorder::dispatchHashCount(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline,
    int photonCount)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashCountPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.dispatch(uint32_t(photonCount) * 16 / 256, 1, 1);
}

void FrameRecorder::dispatchHashScan(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashScanPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.dispatch(1, 1, 1);  // single workgroup of 1024 threads
}

void FrameRecorder::dispatchHashScatter(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline,
    int photonCount)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashScatterPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    cb.dispatch(uint32_t(photonCount) * 16 / 256, 1, 1);
}

void FrameRecorder::dispatchHashAggregate(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashAggregatePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    // HASH_TABLE_SIZE / 256 threads, one per hash cell
    cb.dispatch(1024, 1, 1);
}

void FrameRecorder::dispatchHashGather(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetHashGatherPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    // HASH_TABLE_SIZE / 256 threads, one per hash cell
    cb.dispatch(1024, 1, 1);
}

void FrameRecorder::dispatchStatsOverlay(
    vk::CommandBuffer cb, uint32_t f, RayTracingPipeline& pipeline)
{
    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetStatsOverlayPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
    // Small panel: 35x23 workgroups of 8x8 = 280x184 px
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
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_nrmImg, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_depImg, sub},
        {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
         vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_outImg, sub},
    };
    // Memory barrier: accumData (SSBO binding 13) written by RT, read by denoiser.
    // Image barriers cover output/normal/depth images; global barrier covers buffers.
    vk::MemoryBarrier mb(vk::AccessFlagBits::eShaderWrite,
                         vk::AccessFlagBits::eShaderRead);
    cb.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
        vk::PipelineStageFlagBits::eComputeShader, {}, mb, {}, gb);

    cb.bindPipeline(vk::PipelineBindPoint::eCompute,
                    pipeline.GetDenoisePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline.Desc().PipelineLayout(), 0,
        pipeline.Desc().DescriptorSet(f), nullptr);
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

    vk::Image sw = m_swapImages[imageIndex];

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
    vk::PipelineStageFlags ws = vk::PipelineStageFlagBits::eRayTracingShaderKHR;
    m_device.getQueue(m_cqf, 0)
        .submit(vk::SubmitInfo(imageAvail, ws, cb, renderDone), fence);
    m_device.getQueue(m_pqf, 0)
        .presentKHR(vk::PresentInfoKHR(renderDone, *m_swapchain, imageIndex));

    if (first)
        m_device.waitForFences(fence, true, UINT64_MAX);

    if (m_hasTS && !first) {
        uint64_t r[2];
        auto res = static_cast<vk::Device>(m_device).getQueryPoolResults(
            *m_tsPool, f * TS_PER_FRAME, TS_PER_FRAME,
            sizeof(r), r, sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (res == vk::Result::eSuccess && r[0] > 0 && r[1] > r[0]) {
            float ms = float(r[1] - r[0]) * m_tsPeriod / 1e6f;
            m_lastGpuMs = ms;
            m_frameCount++;
            if (m_frameCount % 60 == 0)
                Log::info("[GPU] frame {} compute time: {} ms",
                          m_frameCount, ms);
        }
    }
}
