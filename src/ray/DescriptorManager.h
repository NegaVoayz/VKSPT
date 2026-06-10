#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <vector>

/// Owns descriptor set layout, pool, and per-frame descriptor sets.
/// Provides bind methods for all 17 shader bindings (0–16).
class DescriptorManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    explicit DescriptorManager(const vk::raii::Device& device);

    vk::PipelineLayout       pipelineLayout() const;
    vk::DescriptorSet        descriptorSet(uint32_t i) const;

    // Bind methods — one per binding
    void bindTLAS(uint32_t fi, vk::AccelerationStructureKHR tlas);
    void bindOutputImage(uint32_t fi, vk::ImageView v, vk::Sampler s);
    void bindMaterialBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindLightBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindGeometrySSBOs(uint32_t fi,
        vk::Buffer vBuf, vk::DeviceSize vSz,
        vk::Buffer iBuf, vk::DeviceSize iSz,
        vk::Buffer rBuf, vk::DeviceSize rSz);
    void bindRayBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindCounterBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindPixelAccum(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindOverflowBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindEnvMap(uint32_t fi, vk::ImageView v, vk::Sampler s);
    void bindNormalSSBO(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindAccumBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);
    void bindNormalImage(uint32_t fi, vk::ImageView v);
    void bindDepthImage(uint32_t fi, vk::ImageView v);
    void bindInstanceNormalBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz);

private:
    void createLayout();
    void createPool();
    void allocateSets();

    const vk::raii::Device& m_device;
    vk::raii::DescriptorSetLayout m_layout = nullptr;
    vk::raii::PipelineLayout      m_pipeLayout = nullptr;
    vk::raii::DescriptorPool      m_pool = nullptr;
    std::vector<vk::raii::DescriptorSet> m_sets;
};
