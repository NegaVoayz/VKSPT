#pragma once

#include "core/GPUBuffer.h"
#include "ray/DescriptorManager.h"
#include <vulkan/vulkan_raii.hpp>
#include <string>

class RayTracingPipeline {
public:
    RayTracingPipeline(const vk::raii::Device& d, const vk::raii::PhysicalDevice& pd,
                       uint32_t cqf, uint32_t w, uint32_t h);
    ~RayTracingPipeline();
    RayTracingPipeline(const RayTracingPipeline&) = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;
    RayTracingPipeline(RayTracingPipeline&&) = delete;
    RayTracingPipeline& operator=(RayTracingPipeline&&) = delete;

    // Legacy compute pipelines (denoiser still needs compute)
    void createPipeline(const std::string& spv);
    void createSortPipeline(const std::string& spv);
    void createNormalizePipeline(const std::string& spv);
    void createClassifyPipeline(const std::string& spv);
    void createProcessPipeline(const std::string& spv);
    void createDenoisePipeline(const std::string& spv);

    vk::Pipeline getPipeline() const { return *m_pipeline; }
    vk::Pipeline getSortPipeline() const { return *m_sortPipe; }
    vk::Pipeline getNormalizePipeline() const { return *m_normPipe; }
    vk::Pipeline getClassifyPipeline() const { return *m_classPipe; }
    vk::Pipeline getProcessPipeline() const { return *m_procPipe; }
    vk::Pipeline getDenoisePipeline() const { return *m_denPipe; }

    // RT pipeline
    void createRTPipeline(const std::string& spv);
    vk::Pipeline getRTPipeline() const { return *m_rtPipeline; }
    const vk::StridedDeviceAddressRegionKHR& raygenRegion()  const { return m_raygenRegion; }
    const vk::StridedDeviceAddressRegionKHR& missRegion()    const { return m_missRegion; }
    const vk::StridedDeviceAddressRegionKHR& hitRegion()     const { return m_hitRegion; }
    const vk::StridedDeviceAddressRegionKHR& callableRegion() const { return m_callableRegion; }

    DescriptorManager& desc() { return m_desc; }
    const DescriptorManager& desc() const { return m_desc; }

private:
    static std::vector<uint32_t> readFile(const std::string& p);
    vk::raii::Pipeline mkPipe(vk::raii::ShaderModule& m, const std::string& s);

    const vk::raii::Device&         m_dev;
    const vk::raii::PhysicalDevice& m_physDev;
    uint32_t                        m_computeQf;
    DescriptorManager m_desc;

    // Compute pipelines
    vk::raii::ShaderModule m_sm = nullptr;
    vk::raii::Pipeline     m_pipeline = nullptr;
    vk::raii::ShaderModule m_sortSm = nullptr;
    vk::raii::Pipeline     m_sortPipe = nullptr;
    vk::raii::ShaderModule m_normSm = nullptr;
    vk::raii::Pipeline     m_normPipe = nullptr;
    vk::raii::ShaderModule m_classSm = nullptr;
    vk::raii::Pipeline     m_classPipe = nullptr;
    vk::raii::ShaderModule m_procSm = nullptr;
    vk::raii::Pipeline     m_procPipe = nullptr;
    vk::raii::ShaderModule m_denSm = nullptr;
    vk::raii::Pipeline     m_denPipe = nullptr;

    // RT pipeline
    vk::raii::ShaderModule               m_rtSm = nullptr;
    vk::raii::Pipeline                   m_rtPipeline = nullptr;
    GPUBuffer                            m_sbtBuffer;
    uint32_t                             m_sbtHandleSize = 0;
    uint32_t                             m_sbtStride = 0;
    vk::StridedDeviceAddressRegionKHR    m_raygenRegion{};
    vk::StridedDeviceAddressRegionKHR    m_missRegion{};
    vk::StridedDeviceAddressRegionKHR    m_hitRegion{};
    vk::StridedDeviceAddressRegionKHR    m_callableRegion{};
};
