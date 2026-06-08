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
    allocateDescriptorSets();
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
    vk::DescriptorSetLayoutBinding tlasBinding(
        0, vk::DescriptorType::eAccelerationStructureKHR,
        1, vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutBinding imageBinding(
        1, vk::DescriptorType::eStorageImage,
        1, vk::ShaderStageFlagBits::eCompute
    );
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {tlasBinding, imageBinding};

    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
    m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, layoutInfo);
}

void RayTracingPipeline::createPipelineLayout() {
    vk::PushConstantRange pushRange(
        vk::ShaderStageFlagBits::eCompute, 0, 64
    );
    vk::PipelineLayoutCreateInfo layoutInfo({}, *m_descriptorSetLayout, pushRange);
    m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);
}

void RayTracingPipeline::createDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT},
        {vk::DescriptorType::eStorageImage,             MAX_FRAMES_IN_FLIGHT},
    };
    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        MAX_FRAMES_IN_FLIGHT, poolSizes
    );
    m_descriptorPool = vk::raii::DescriptorPool(m_device, poolInfo);
}

void RayTracingPipeline::allocateDescriptorSets() {
    m_descriptorSets.reserve(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorSetAllocateInfo allocInfo(*m_descriptorPool, *m_descriptorSetLayout);
        auto sets = vk::raii::DescriptorSets(m_device, allocInfo);
        m_descriptorSets.emplace_back(std::move(sets[0]));
    }
}

// -----------------------------------------------------------------------------
// SPIR-V Loading & Pipeline Creation
// -----------------------------------------------------------------------------
static std::vector<uint32_t> readShaderFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    // SPIR-V is always a multiple of 4 bytes; round up to be safe
    size_t wordCount = (fileSize + 3) / 4;
    std::vector<uint32_t> buffer(wordCount);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(fileSize));
    file.close();
    return buffer;
}

void RayTracingPipeline::createPipeline(const std::string& spirvPath) {
    auto shaderCode = readShaderFile(spirvPath);
    vk::ShaderModuleCreateInfo shaderInfo(
        {}, shaderCode.size() * sizeof(uint32_t),
        shaderCode.data()
    );
    m_shaderModule = vk::raii::ShaderModule(m_device, shaderInfo);

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
void RayTracingPipeline::bindTLAS(uint32_t frameIndex, vk::AccelerationStructureKHR tlas) {
    vk::WriteDescriptorSetAccelerationStructureKHR asInfo(1, &tlas);
    vk::WriteDescriptorSet write(
        *m_descriptorSets[frameIndex], 0, 0, 1,
        vk::DescriptorType::eAccelerationStructureKHR,
        nullptr, nullptr, nullptr, &asInfo
    );
    m_device.updateDescriptorSets(write, nullptr);
}

void RayTracingPipeline::bindOutputImage(uint32_t frameIndex, vk::ImageView imageView,
                                          vk::Sampler sampler) {
    vk::DescriptorImageInfo imageInfo(sampler, imageView, vk::ImageLayout::eGeneral);
    vk::WriteDescriptorSet write(
        *m_descriptorSets[frameIndex], 1, 0,
        vk::DescriptorType::eStorageImage,
        imageInfo
    );
    m_device.updateDescriptorSets(write, nullptr);
}
