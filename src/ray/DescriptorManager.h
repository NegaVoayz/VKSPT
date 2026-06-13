#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <vector>

/// Owns descriptor set layout, pool, and per-frame descriptor sets.
/// Provides bind methods for all shader bindings (0–6, 11–16).
class DescriptorManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    explicit DescriptorManager(const vk::raii::Device& device);

    vk::PipelineLayout       PipelineLayout() const;
    vk::DescriptorSet        DescriptorSet(uint32_t i) const;

    // Bind methods — one per binding
    void BindTLAS(uint32_t fi, vk::AccelerationStructureKHR tlas);
    void BindOutputImage(uint32_t fi, vk::ImageView v, vk::Sampler s);
    void BindMaterialBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindLightBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindGeometrySSBOs(uint32_t fi,
        vk::Buffer vBuf, vk::DeviceSize vSz,
        vk::Buffer iBuf, vk::DeviceSize iSz,
        vk::Buffer rBuf, vk::DeviceSize rSz);
    void BindEnvMap(uint32_t fi, vk::ImageView v, vk::Sampler s);
    void BindNormalSSBO(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindAccumBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindNormalImage(uint32_t fi, vk::ImageView v);
    void BindDepthImage(uint32_t fi, vk::ImageView v);
    void BindInstanceNormalBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindPhotonBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindPhotonCounter(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindHashCellData(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindSortedPhotonIndices(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindCellPhotonData(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void BindRayStats(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void createLayout();
    void createPool();
    void allocateSets();

    const vk::raii::Device& m_device;
    vk::raii::DescriptorSetLayout m_layout = nullptr;
    vk::raii::PipelineLayout      m_pipeLayout = nullptr;
    vk::raii::DescriptorPool      m_pool = nullptr;
    std::vector<vk::raii::DescriptorSet> m_sets;
};
