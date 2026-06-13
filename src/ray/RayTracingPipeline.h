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

    // Denoiser (compute pipeline)
    void CreateDenoisePipeline(const std::string& spv);
    vk::Pipeline GetDenoisePipeline() const { return *m_denPipe; }

    // Hash grid builders (compute pipelines)
    void CreateHashCountPipeline(const std::string& spv);
    void CreateHashScanPipeline(const std::string& spv);
    void CreateHashScatterPipeline(const std::string& spv);
    vk::Pipeline GetHashCountPipeline()   const { return *m_hashCountPipe; }
    vk::Pipeline GetHashScanPipeline()    const { return *m_hashScanPipe; }
    vk::Pipeline GetHashScatterPipeline() const { return *m_hashScatterPipe; }

    // Hash aggregate (compute — one representative photon per cell)
    void CreateHashAggregatePipeline(const std::string& spv);
    vk::Pipeline GetHashAggregatePipeline() const { return *m_hashAggregatePipe; }

    // Stats overlay (compute pipeline)
    void CreateStatsOverlayPipeline(const std::string& spv);
    vk::Pipeline GetStatsOverlayPipeline() const { return *m_statsOverlayPipe; }

    // RT pipeline
    void CreateRTPipeline(const std::string& spv);
    vk::Pipeline GetRTPipeline() const { return *m_rtPipeline; }
    const vk::StridedDeviceAddressRegionKHR& RaygenRegion()        const { return m_raygenRegion; }
    const vk::StridedDeviceAddressRegionKHR& PhotonRaygenRegion()  const { return m_photonRaygenRegion; }
    const vk::StridedDeviceAddressRegionKHR& MissRegion()          const { return m_missRegion; }
    const vk::StridedDeviceAddressRegionKHR& HitRegion()           const { return m_hitRegion; }
    const vk::StridedDeviceAddressRegionKHR& CallableRegion()      const { return m_callableRegion; }

    DescriptorManager& Desc() { return m_desc; }
    const DescriptorManager& Desc() const { return m_desc; }

private:
    static std::vector<uint32_t> readFile(const std::string& p);
    vk::raii::Pipeline mkPipe(vk::raii::ShaderModule& m, const std::string& s);
    void querySbtProperties();
    std::vector<uint8_t> getShaderGroupHandles() const;
    void uploadSbtBuffer(const std::vector<uint8_t>& handles);

    const vk::raii::Device&         m_dev;
    const vk::raii::PhysicalDevice& m_physDev;
    uint32_t                        m_computeQf;
    DescriptorManager m_desc;

    // Compute pipeline (denoiser)
    vk::raii::ShaderModule m_denSm = nullptr;
    vk::raii::Pipeline     m_denPipe = nullptr;

    // Compute pipelines (hash grid builders)
    vk::raii::ShaderModule m_hashCountSm = nullptr;
    vk::raii::Pipeline     m_hashCountPipe = nullptr;
    vk::raii::ShaderModule m_hashScanSm = nullptr;
    vk::raii::Pipeline     m_hashScanPipe = nullptr;
    vk::raii::ShaderModule m_hashScatterSm = nullptr;
    vk::raii::Pipeline     m_hashScatterPipe = nullptr;
    vk::raii::ShaderModule m_hashAggregateSm = nullptr;
    vk::raii::Pipeline     m_hashAggregatePipe = nullptr;

    // Stats overlay compute pipeline
    vk::raii::ShaderModule m_statsOverlaySm = nullptr;
    vk::raii::Pipeline     m_statsOverlayPipe = nullptr;

    // RT pipeline
    vk::raii::ShaderModule               m_rtSm = nullptr;
    vk::raii::Pipeline                   m_rtPipeline = nullptr;
    GPUBuffer                            m_sbtBuffer;
    uint32_t                             m_sbtHandleSize = 0;
    uint32_t                             m_sbtStride = 0;
    vk::StridedDeviceAddressRegionKHR    m_raygenRegion{};
    vk::StridedDeviceAddressRegionKHR    m_photonRaygenRegion{};
    vk::StridedDeviceAddressRegionKHR    m_missRegion{};
    vk::StridedDeviceAddressRegionKHR    m_hitRegion{};
    vk::StridedDeviceAddressRegionKHR    m_callableRegion{};
};
