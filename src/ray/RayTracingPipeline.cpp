#include "ray/RayTracingPipeline.h"
#include "core/Log.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

static constexpr uint32_t RT_GROUP_COUNT = 4;  // raygen(camera), chit, miss, raygen(photon)

RayTracingPipeline::RayTracingPipeline(
    const vk::raii::Device& d, const vk::raii::PhysicalDevice& pd,
    uint32_t cqf, uint32_t, uint32_t)
    : m_dev(d), m_physDev(pd), m_desc(d), m_computeQf(cqf) {}
RayTracingPipeline::~RayTracingPipeline() = default;

std::vector<uint32_t> RayTracingPipeline::readFile(const std::string& p) {
    std::ifstream f(p, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Failed: " + p);
    size_t sz = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> buf((sz + 3) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(sz));
    return buf;
}

vk::raii::Pipeline RayTracingPipeline::mkPipe(
    vk::raii::ShaderModule& m, const std::string& s)
{
    auto code = readFile(s);
    m = vk::raii::ShaderModule(m_dev,
        vk::ShaderModuleCreateInfo({}, code.size()*4, code.data()));
    vk::PipelineShaderStageCreateInfo stage(
        {}, vk::ShaderStageFlagBits::eCompute, *m, "main");
    return vk::raii::Pipeline(m_dev, nullptr,
        vk::ComputePipelineCreateInfo({}, stage, m_desc.PipelineLayout()));
}

void RayTracingPipeline::CreateDenoisePipeline(const std::string& s)
    { m_denPipe = mkPipe(m_denSm, s); }

void RayTracingPipeline::CreateHashCountPipeline(const std::string& s)
    { m_hashCountPipe = mkPipe(m_hashCountSm, s); }

void RayTracingPipeline::CreateHashScanPipeline(const std::string& s)
    { m_hashScanPipe = mkPipe(m_hashScanSm, s); }

void RayTracingPipeline::CreateHashScatterPipeline(const std::string& s)
    { m_hashScatterPipe = mkPipe(m_hashScatterSm, s); }

// ---- RT Pipeline + SBT ----
void RayTracingPipeline::CreateRTPipeline(const std::string& spv)
{
    auto code = readFile(spv);
    m_rtSm = vk::raii::ShaderModule(m_dev,
        vk::ShaderModuleCreateInfo({}, code.size() * 4, code.data()));

    vk::PipelineShaderStageCreateInfo stages[RT_GROUP_COUNT] = {
        {{}, vk::ShaderStageFlagBits::eRaygenKHR,      *m_rtSm, "RayGenMain"},
        {{}, vk::ShaderStageFlagBits::eClosestHitKHR,   *m_rtSm, "ClosestHitMain"},
        {{}, vk::ShaderStageFlagBits::eMissKHR,         *m_rtSm, "MissMain"},
        {{}, vk::ShaderStageFlagBits::eRaygenKHR,       *m_rtSm, "PhotonRayGenMain"},
    };

    vk::RayTracingShaderGroupCreateInfoKHR groups[RT_GROUP_COUNT] = {
        {vk::RayTracingShaderGroupTypeKHR::eGeneral,
         0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
         VK_SHADER_UNUSED_KHR, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {vk::RayTracingShaderGroupTypeKHR::eGeneral,
         2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
        {vk::RayTracingShaderGroupTypeKHR::eGeneral,
         3, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
    };

    vk::RayTracingPipelineCreateInfoKHR rtInfo;
    rtInfo.setStages(stages);
    rtInfo.setGroups(groups);
    rtInfo.setMaxPipelineRayRecursionDepth(31);
    rtInfo.setLayout(m_desc.PipelineLayout());

    auto rawDev = static_cast<VkDevice>(*m_dev);
    VkPipeline rawPipeline;
    auto result = static_cast<vk::Result>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateRayTracingPipelinesKHR(
            rawDev, nullptr, nullptr, 1,
            reinterpret_cast<const VkRayTracingPipelineCreateInfoKHR*>(&rtInfo),
            nullptr, &rawPipeline));
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to create RT pipeline");
    m_rtPipeline = vk::raii::Pipeline(m_dev, rawPipeline);

    querySbtProperties();
    auto handles = getShaderGroupHandles();
    uploadSbtBuffer(handles);
}

void RayTracingPipeline::querySbtProperties()
{
    auto rtProps = m_physDev.getProperties2<
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR
    >().get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    m_sbtHandleSize = rtProps.shaderGroupHandleSize;
    uint32_t align = std::max(rtProps.shaderGroupBaseAlignment,
                              rtProps.shaderGroupHandleSize);
    m_sbtStride = ((m_sbtHandleSize + align - 1) / align) * align;
}

std::vector<uint8_t> RayTracingPipeline::getShaderGroupHandles() const
{
    uint32_t dataSize = RT_GROUP_COUNT * m_sbtHandleSize;
    std::vector<uint8_t> handles(dataSize);
    auto result = static_cast<vk::Result>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetRayTracingShaderGroupHandlesKHR(
            static_cast<VkDevice>(*m_dev), *m_rtPipeline, 0, RT_GROUP_COUNT,
            dataSize, handles.data()));
    if (result != vk::Result::eSuccess)
        throw std::runtime_error("Failed to get RT shader group handles");
    return handles;
}

void RayTracingPipeline::uploadSbtBuffer(const std::vector<uint8_t>& handles)
{
    vk::DeviceSize sbtSize = m_sbtStride * RT_GROUP_COUNT;
    std::vector<uint8_t> sbtData(static_cast<size_t>(sbtSize), 0);
    for (uint32_t i = 0; i < RT_GROUP_COUNT; ++i)
        std::memcpy(sbtData.data() + i * m_sbtStride,
                    handles.data() + i * m_sbtHandleSize,
                    m_sbtHandleSize);

    auto staging = GPUBuffer::CreateStaging(m_dev, sbtData.data(), sbtSize,
        vk::BufferUsageFlagBits::eTransferSrc, m_physDev);
    m_sbtBuffer = GPUBuffer::Create(m_dev, sbtSize,
        vk::BufferUsageFlagBits::eShaderBindingTableKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDev);

    {
        vk::raii::CommandPool cmdPool(m_dev, {{}, m_computeQf});
        vk::raii::CommandBuffers cmdBufs(m_dev,
            {*cmdPool, vk::CommandBufferLevel::ePrimary, 1});
        cmdBufs[0].begin(vk::CommandBufferBeginInfo(
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        vk::BufferCopy copy(0, 0, sbtSize);
        cmdBufs[0].copyBuffer(*staging.Buffer, *m_sbtBuffer.Buffer, copy);
        cmdBufs[0].end();
        vk::SubmitInfo si;
        si.setCommandBuffers(*cmdBufs[0]);
        m_dev.getQueue(m_computeQf, 0).submit(si, nullptr);
        m_dev.waitIdle();
    }

    vk::DeviceAddress sbtAddr = m_sbtBuffer.Address;
    m_raygenRegion        = vk::StridedDeviceAddressRegionKHR(sbtAddr + 0 * m_sbtStride, m_sbtStride, m_sbtStride);
    m_hitRegion           = vk::StridedDeviceAddressRegionKHR(sbtAddr + 1 * m_sbtStride, m_sbtStride, m_sbtStride);
    m_missRegion          = vk::StridedDeviceAddressRegionKHR(sbtAddr + 2 * m_sbtStride, m_sbtStride, m_sbtStride);
    m_photonRaygenRegion  = vk::StridedDeviceAddressRegionKHR(sbtAddr + 3 * m_sbtStride, m_sbtStride, m_sbtStride);
    m_callableRegion      = vk::StridedDeviceAddressRegionKHR(0, 0, 0);

    Log::info("RT pipeline created: {} shader groups, SBT {} bytes", RT_GROUP_COUNT, sbtSize);
}
