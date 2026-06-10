#include "ray/RayTracingPipeline.h"
#include <fstream>
#include <stdexcept>

RayTracingPipeline::RayTracingPipeline(
    const vk::raii::Device& d, uint32_t, uint32_t)
    : m_dev(d), m_desc(d) {}
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
        vk::ComputePipelineCreateInfo({}, stage, m_desc.pipelineLayout()));
}

void RayTracingPipeline::createPipeline(const std::string& s)
    { m_pipeline = mkPipe(m_sm, s); }
void RayTracingPipeline::createSortPipeline(const std::string& s)
    { m_sortPipe = mkPipe(m_sortSm, s); }
void RayTracingPipeline::createNormalizePipeline(const std::string& s)
    { m_normPipe = mkPipe(m_normSm, s); }
void RayTracingPipeline::createClassifyPipeline(const std::string& s)
    { m_classPipe = mkPipe(m_classSm, s); }
void RayTracingPipeline::createProcessPipeline(const std::string& s)
    { m_procPipe = mkPipe(m_procSm, s); }
void RayTracingPipeline::createDenoisePipeline(const std::string& s)
    { m_denPipe = mkPipe(m_denSm, s); }
