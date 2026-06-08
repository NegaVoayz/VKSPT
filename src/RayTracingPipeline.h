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

    /// Bind the TLAS acceleration structure for the given frame index.
    void bindTLAS(uint32_t frameIndex, vk::AccelerationStructureKHR tlas);

    /// Bind the output storage image for the given frame index.
    void bindOutputImage(uint32_t frameIndex, vk::ImageView imageView, vk::Sampler sampler);

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
};
