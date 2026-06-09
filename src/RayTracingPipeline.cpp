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
//   binding 2: uniform buffer (material array)
//   binding 3: uniform buffer (light SPD array)
//   binding 4: storage buffer (vertex data — float[] for geometric normals)
//   binding 5: storage buffer (index data — uint[] for geometric normals)
//   binding 6: storage buffer (instance ranges — SoA: vtxOffset/vtxCount/idxOffset/idxCount/materialID arrays)
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
    vk::DescriptorSetLayoutBinding materialBinding(
        2, vk::DescriptorType::eUniformBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutBinding lightBinding(
        3, vk::DescriptorType::eUniformBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutBinding vertexBinding(
        4, vk::DescriptorType::eStorageBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutBinding indexBinding(
        5, vk::DescriptorType::eStorageBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutBinding rangeBinding(
        6, vk::DescriptorType::eStorageBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    // Phase 4 sorted pipeline: global ray buffer + counters
    vk::DescriptorSetLayoutBinding rayBufBinding(
        7, vk::DescriptorType::eStorageBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutBinding counterBinding(
        8, vk::DescriptorType::eStorageBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutBinding accumBinding(
        9, vk::DescriptorType::eStorageBuffer,
        1, vk::ShaderStageFlagBits::eCompute
    );
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        tlasBinding, imageBinding, materialBinding, lightBinding,
        vertexBinding, indexBinding, rangeBinding,
        rayBufBinding, counterBinding, accumBinding
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
    m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_device, layoutInfo);
}

void RayTracingPipeline::createPipelineLayout() {
    vk::PushConstantRange pushRange(
        vk::ShaderStageFlagBits::eCompute, 0, 112
    );
    vk::PipelineLayoutCreateInfo layoutInfo({}, *m_descriptorSetLayout, pushRange);
    m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);
}

void RayTracingPipeline::createDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT},
        {vk::DescriptorType::eStorageImage,             MAX_FRAMES_IN_FLIGHT},
        {vk::DescriptorType::eUniformBuffer,            MAX_FRAMES_IN_FLIGHT * 2},
        {vk::DescriptorType::eStorageBuffer,            MAX_FRAMES_IN_FLIGHT * 6},
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

void RayTracingPipeline::bindMaterialBuffer(uint32_t frameIndex, vk::Buffer buffer,
                                             vk::DeviceSize size) {
    vk::DescriptorBufferInfo bufferInfo(buffer, 0, size);
    vk::WriteDescriptorSet write(
        *m_descriptorSets[frameIndex], 2, 0, 1,
        vk::DescriptorType::eUniformBuffer,
        nullptr, &bufferInfo
    );
    m_device.updateDescriptorSets(write, nullptr);
}

void RayTracingPipeline::bindLightBuffer(uint32_t frameIndex, vk::Buffer buffer,
                                          vk::DeviceSize size) {
    vk::DescriptorBufferInfo bufferInfo(buffer, 0, size);
    vk::WriteDescriptorSet write(
        *m_descriptorSets[frameIndex], 3, 0, 1,
        vk::DescriptorType::eUniformBuffer,
        nullptr, &bufferInfo
    );
    m_device.updateDescriptorSets(write, nullptr);
}

void RayTracingPipeline::bindGeometrySSBOs(uint32_t frameIndex,
                                            vk::Buffer vertexBuf, vk::DeviceSize vertexSize,
                                            vk::Buffer indexBuf,  vk::DeviceSize indexSize,
                                            vk::Buffer rangeBuf,  vk::DeviceSize rangeSize) {
    std::vector<vk::WriteDescriptorSet> writes;

    vk::DescriptorBufferInfo vertexInfo(vertexBuf, 0, vertexSize);
    writes.emplace_back(*m_descriptorSets[frameIndex], 4, 0, 1,
                        vk::DescriptorType::eStorageBuffer, nullptr, &vertexInfo);

    vk::DescriptorBufferInfo indexInfo(indexBuf, 0, indexSize);
    writes.emplace_back(*m_descriptorSets[frameIndex], 5, 0, 1,
                        vk::DescriptorType::eStorageBuffer, nullptr, &indexInfo);

    vk::DescriptorBufferInfo rangeInfo(rangeBuf, 0, rangeSize);
    writes.emplace_back(*m_descriptorSets[frameIndex], 6, 0, 1,
                        vk::DescriptorType::eStorageBuffer, nullptr, &rangeInfo);

    m_device.updateDescriptorSets(writes, nullptr);
}

void RayTracingPipeline::bindRayBuffer(uint32_t frameIndex, vk::Buffer buf,
                                        vk::DeviceSize size) {
    vk::DescriptorBufferInfo info(buf, 0, size);
    vk::WriteDescriptorSet write(
        *m_descriptorSets[frameIndex], 7, 0, 1,
        vk::DescriptorType::eStorageBuffer, nullptr, &info
    );
    m_device.updateDescriptorSets(write, nullptr);
}

void RayTracingPipeline::bindCounterBuffer(uint32_t frameIndex, vk::Buffer buf,
                                            vk::DeviceSize size) {
    vk::DescriptorBufferInfo info(buf, 0, size);
    vk::WriteDescriptorSet write(
        *m_descriptorSets[frameIndex], 8, 0, 1,
        vk::DescriptorType::eStorageBuffer, nullptr, &info
    );
    m_device.updateDescriptorSets(write, nullptr);
}

void RayTracingPipeline::bindPixelAccum(uint32_t frameIndex, vk::Buffer buf,
                                         vk::DeviceSize size) {
    vk::DescriptorBufferInfo info(buf, 0, size);
    vk::WriteDescriptorSet write(
        *m_descriptorSets[frameIndex], 9, 0, 1,
        vk::DescriptorType::eStorageBuffer, nullptr, &info
    );
    m_device.updateDescriptorSets(write, nullptr);
}

void RayTracingPipeline::createSortPipeline(const std::string& spirvPath) {
    auto shaderCode = readShaderFile(spirvPath);
    vk::ShaderModuleCreateInfo shaderInfo(
        {}, shaderCode.size() * sizeof(uint32_t), shaderCode.data()
    );
    m_sortShaderModule = vk::raii::ShaderModule(m_device, shaderInfo);

    vk::PipelineShaderStageCreateInfo stageInfo(
        {}, vk::ShaderStageFlagBits::eCompute, *m_sortShaderModule, "main"
    );
    vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, *m_pipelineLayout);
    m_sortPipeline = vk::raii::Pipeline(m_device, nullptr, pipelineInfo);
}
