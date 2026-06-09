#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <string>
#include <vector>

/// Manages descriptor sets, pipeline layout, and the ray tracing compute pipeline.
/// Binds TLAS + output storage image for the compute shader.
class RayTracingPipeline {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    RayTracingPipeline(
        const vk::raii::Device& device,
        uint32_t                width,
        uint32_t                height
    );
    ~RayTracingPipeline();

    // Non-copyable, non-movable
    RayTracingPipeline(const RayTracingPipeline&)            = delete;
    RayTracingPipeline& operator=(const RayTracingPipeline&) = delete;
    RayTracingPipeline(RayTracingPipeline&&)                 = delete;
    RayTracingPipeline& operator=(RayTracingPipeline&&)      = delete;

    /// Load SPIR-V shader and create the compute pipeline.
    void createPipeline(const std::string& spirvPath);

    /// Phase 4: Load sorted pipeline shader and create a second compute pipeline.
    void createSortPipeline(const std::string& spirvPath);

    /// Get sorted pipeline.
    vk::Pipeline getSortPipeline() const { return *m_sortPipeline; }

    /// Bind the TLAS acceleration structure for the given frame index.
    void bindTLAS(uint32_t frameIndex, vk::AccelerationStructureKHR tlas);

    /// Bind the output storage image for the given frame index.
    void bindOutputImage(uint32_t frameIndex, vk::ImageView imageView, vk::Sampler sampler);

    /// Bind the material uniform buffer for the given frame index.
    void bindMaterialBuffer(uint32_t frameIndex, vk::Buffer buffer, vk::DeviceSize size);

    /// Bind the light uniform buffer for the given frame index.
    void bindLightBuffer(uint32_t frameIndex, vk::Buffer buffer, vk::DeviceSize size);

    /// Bind geometry SSBOs (vertex data, index data, instance ranges) for shader normals.
    void bindGeometrySSBOs(uint32_t frameIndex,
                           vk::Buffer vertexBuf, vk::DeviceSize vertexSize,
                           vk::Buffer indexBuf,  vk::DeviceSize indexSize,
                           vk::Buffer rangeBuf,  vk::DeviceSize rangeSize);

    /// Phase 4 sorted pipeline: bind global ray buffer (binding=7).
    void bindRayBuffer(uint32_t frameIndex, vk::Buffer buf, vk::DeviceSize size);

    /// Phase 4 sorted pipeline: bind action counter buffer (binding=8).
    void bindCounterBuffer(uint32_t frameIndex, vk::Buffer buf, vk::DeviceSize size);

    /// Phase 4 sorted pipeline: bind pixel accumulator buffer (binding=9).
    void bindPixelAccum(uint32_t frameIndex, vk::Buffer buf, vk::DeviceSize size);

    /// Get the descriptor set for the given frame index.
    vk::DescriptorSet getDescriptorSet(uint32_t frameIndex) const {
        return *m_descriptorSets[frameIndex];
    }

    /// Get pipeline layout.
    vk::PipelineLayout getPipelineLayout() const { return *m_pipelineLayout; }

    /// Get compute pipeline.
    vk::Pipeline getPipeline() const { return *m_pipeline; }

private:
    void createDescriptorSetLayout();
    void createPipelineLayout();
    void createDescriptorPool();
    void allocateDescriptorSets();

    const vk::raii::Device& m_device;
    uint32_t                m_width;
    uint32_t                m_height;

    vk::raii::DescriptorSetLayout                    m_descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout                         m_pipelineLayout      = nullptr;
    vk::raii::DescriptorPool                         m_descriptorPool      = nullptr;
    std::vector<vk::raii::DescriptorSet> m_descriptorSets;  // one per frame-in-flight
    vk::raii::ShaderModule                           m_shaderModule        = nullptr;
    vk::raii::Pipeline                               m_pipeline            = nullptr;
    vk::raii::ShaderModule                           m_sortShaderModule    = nullptr;
    vk::raii::Pipeline                               m_sortPipeline        = nullptr;
};
