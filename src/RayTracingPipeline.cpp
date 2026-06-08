#include "RayTracingPipeline.h"

#include <fstream>
#include <stdexcept>
#include <vector>

RayTracingPipeline::RayTracingPipeline(
    const vk::raii::Device& device,
    uint32_t                width,
    uint32_t                height)
    : m_device(device)
    , m_width(width)
    , m_height(height)
{
    createDescriptorSetLayout();
    createPipelineLayout();
    createDescriptorPool();
    allocateDescriptorSet();
}

RayTracingPipeline::~RayTracingPipeline() {
    // vk::raii handles cleanup
}

// -----------------------------------------------------------------------------
// Descriptor Set Layout
//   binding 0: acceleration structure (TLAS)
//   binding 1: storage image (rgba8 output)
// -----------------------------------------------------------------------------
void RayTracingPipeline::createDescriptorSetLayout() {
    // Binding 0: TLAS acceleration structure
    vk::DescriptorSetLayoutBinding tlasBinding(
        0, vk::DescriptorType::eAccelerationStructureKHR,
        1, vk::ShaderStageFlagBits::eCompute
    );

    // Binding 1: output storage image
    vk::DescriptorSetLayoutBinding imageBinding(
        1, vk::DescriptorType::eStorageImage,
        1, vk::ShaderStageFlagBits::eCompute
    );

    std::vector<vk::DescriptorSetLayoutBinding> bindings = {tlasBinding, imageBinding};

    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
    m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, layoutInfo);
}

void RayTracingPipeline::createPipelineLayout() {
    vk::PipelineLayoutCreateInfo layoutInfo({}, *m_descriptorSetLayout);
    m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);
}

void RayTracingPipeline::createDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eAccelerationStructureKHR, 1},
        {vk::DescriptorType::eStorageImage,             1},
    };

    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes
    );
    m_descriptorPool = vk::raii::DescriptorPool(m_device, poolInfo);
}

void RayTracingPipeline::allocateDescriptorSet() {
    vk::DescriptorSetAllocateInfo allocInfo(*m_descriptorPool, *m_descriptorSetLayout);
    auto sets = vk::raii::DescriptorSets(m_device, allocInfo);
    m_descriptorSet = std::move(sets[0]);
}

// -----------------------------------------------------------------------------
// SPIR-V Loading & Pipeline Creation
// -----------------------------------------------------------------------------
static std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    file.close();
    return buffer;
}

void RayTracingPipeline::createPipeline(const std::string& spirvPath) {
    auto shaderCode = readFile(spirvPath);

    vk::ShaderModuleCreateInfo shaderInfo(
        {}, shaderCode.size(),
        reinterpret_cast<const uint32_t*>(shaderCode.data())
    );
    m_shaderModule = vk::raii::ShaderModule(m_device, shaderInfo);

    // Compute pipeline
    vk::PipelineShaderStageCreateInfo stageInfo(
        {}, vk::ShaderStageFlagBits::eCompute,
        *m_shaderModule, "main"
    );

    vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, *m_pipelineLayout);
    m_pipeline = vk::raii::Pipeline(m_device, nullptr, pipelineInfo);
}

// -----------------------------------------------------------------------------
// Descriptor Updates
// -----------------------------------------------------------------------------
void RayTracingPipeline::bindTLAS(vk::AccelerationStructureKHR tlas) {
    vk::WriteDescriptorSetAccelerationStructureKHR asInfo(1, &tlas);

    vk::WriteDescriptorSet write(
        *m_descriptorSet, 0, 0, 1,
        vk::DescriptorType::eAccelerationStructureKHR,
        nullptr, nullptr, nullptr, &asInfo
    );
    m_device.updateDescriptorSets(write, nullptr);
}

void RayTracingPipeline::bindOutputImage(vk::ImageView imageView, vk::Sampler sampler) {
    vk::DescriptorImageInfo imageInfo(sampler, imageView, vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet write(
        *m_descriptorSet, 1, 0,
        vk::DescriptorType::eStorageImage,
        imageInfo
    );
    m_device.updateDescriptorSets(write, nullptr);
}
