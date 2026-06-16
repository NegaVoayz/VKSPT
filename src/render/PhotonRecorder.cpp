#include "render/PhotonRecorder.h"
#include "render/PhotonHashStages.h"
#include "render/FrameRecorder.h"
#include "ray/DescriptorManager.h"
#include "core/Log.h"

PhotonRecorder::PhotonRecorder(
    const vk::raii::Device& device,
    uint32_t                computeQueueFamily)
    : m_device(device)
    , m_computeQueueFamily(computeQueueFamily)
    , m_photonFence(m_device, vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled})
{}

void PhotonRecorder::setupPhotons(
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    vk::ImageView outputView, vk::ImageView normalView, vk::ImageView depthView,
    vk::Buffer accumBuffer, vk::DeviceSize accumSize)
{
    m_photonOutputView = outputView;
    m_photonNormalView = normalView;
    m_photonDepthView  = depthView;
    m_photonAccumBuffer = accumBuffer;
    m_photonAccumSize   = accumSize;

    const auto& cpuLights = as.getLightsCPU();
    int totalLights = static_cast<int>(as.getLightCount());
    float pointIntensity = 0.0f, spotIntensity = 0.0f;
    int   pointCount = 0, spotCount = 0;
    m_activeLightCount = 0;
    for (int i = 0; i < totalLights; i++) {
        int lightType = static_cast<int>(cpuLights[i].pos_type[3]);
        float intensity = cpuLights[i].color_intensity[3];
        if (lightType == 3 || intensity <= 0.0f) continue;
        m_activeLightCount++;
        if (lightType == 0) { pointIntensity += intensity; pointCount++; }
        if (lightType == 2) { spotIntensity += intensity; spotCount++; }
    }
    if (m_activeLightCount == 0) m_activeLightCount = 1;
    m_perLight = m_photonCount / m_activeLightCount;
    Log::info("[Photon] point: {} (I={:.2f}), spot: {} (I={:.2f}), per light: {}",
              pointCount, pointIntensity, spotCount, spotIntensity, m_perLight);

    m_nextLightIndex = 0;

    m_photonCommandPool = vk::raii::CommandPool(m_device,
        {vk::CommandPoolCreateFlagBits::eTransient |
         vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         m_computeQueueFamily});
    auto commandBuffers = vk::raii::CommandBuffers(m_device,
        {*m_photonCommandPool, vk::CommandBufferLevel::ePrimary, 1});
    m_photonCommandBuffer = std::move(commandBuffers[0]);

    m_photonDone = false;

    bindPhotonDescriptors(as, pipeline);
}

void PhotonRecorder::bindPhotonDescriptors(
    const AccelerationStructure& as, RayTracingPipeline& pipeline)
{
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
    pipeline.Desc().BindDisplayCellData(
        ps, *as.getDisplayCellData().Buffer, as.getDisplayCellData().Size);
    pipeline.Desc().BindRayStats(
        ps, *as.getRayStats().Buffer, as.getRayStats().Size);
    pipeline.Desc().BindOutputImage(ps, m_photonOutputView, nullptr);
    pipeline.Desc().BindAccumBuffer(ps, m_photonAccumBuffer, m_photonAccumSize);
    pipeline.Desc().BindNormalImage(ps, m_photonNormalView);
    pipeline.Desc().BindDepthImage(ps, m_photonDepthView);
}

void PhotonRecorder::trySubmitPhotonBatch(
    const AccelerationStructure& as, RayTracingPipeline& pipeline)
{
    if (m_photonDone) return;
    if (m_device.waitForFences(*m_photonFence, false, 0) != vk::Result::eSuccess) return;

    const auto& cpuLights = as.getLightsCPU();
    int totalLights = static_cast<int>(as.getLightCount());
    int batchIndex = -1;
    for (int i = m_nextLightIndex; i < totalLights; i++) {
        if (static_cast<int>(cpuLights[i].pos_type[3]) != 3 &&
            cpuLights[i].color_intensity[3] > 0.0f) {
            batchIndex = i;
            break;
        }
    }

    if (batchIndex < 0) { m_nextLightIndex = 0; m_passCount++; return; }

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
    pipeline.Desc().BindDisplayCellData(
        ps, *as.getDisplayCellData().Buffer, as.getDisplayCellData().Size);
    pipeline.Desc().BindRayStats(
        ps, *as.getRayStats().Buffer, as.getRayStats().Size);

    m_device.resetFences(*m_photonFence);

    m_photonCommandBuffer.reset({});
    recordPhotonBatch(*m_photonCommandBuffer, as, pipeline,
                      batchIndex, m_perLight, m_activeLightCount);

    m_device.getQueue(m_computeQueueFamily, 0)
        .submit(vk::SubmitInfo({}, {}, *m_photonCommandBuffer), *m_photonFence);

    m_nextLightIndex = batchIndex + 1;
    Log::info("[Photon] batch {} done", batchIndex);
}

void PhotonRecorder::recordPhotonBatch(
    vk::CommandBuffer cb,
    const AccelerationStructure& as, RayTracingPipeline& pipeline,
    int lightIndex, int photonsPerBatch, int totalBatches)
{
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cb.fillBuffer(*as.getPhotonCounter().Buffer, 0, 4, 0);
    cb.fillBuffer(*as.getHashCellData().Buffer, 0, as.getHashCellData().Size, 0);

    {
        std::vector<vk::BufferMemoryBarrier> bArr = {
            {vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderWrite,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getPhotonCounter().Buffer, 0, 4},
            {vk::AccessFlagBits::eTransferWrite,
             vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getHashCellData().Buffer, 0, as.getHashCellData().Size},
        };
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR |
                vk::PipelineStageFlagBits::eComputeShader, {}, {}, bArr, {});
    }

    PhotonTraceConfig cfg{m_photonMaxBounces, m_minNeighborPhotons,
                           m_maxGatherRadius, m_hashCellSize, m_passCount};
    DispatchPhotonTraceBatch(cb, as, pipeline, cfg,
                             lightIndex, photonsPerBatch, totalBatches);

    {
        std::vector<vk::BufferMemoryBarrier> bArr = {
            {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getPhotonBuffer().Buffer, 0, as.getPhotonBuffer().Size},
            {vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
             *as.getPhotonCounter().Buffer, 0, 4},
        };
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            vk::PipelineStageFlagBits::eComputeShader, {}, {}, bArr, {});
    }

    RecordPhotonHashStages(cb, as, pipeline, photonsPerBatch);

    cb.end();
}
