#include "ray/DescriptorManager.h"

DescriptorManager::DescriptorManager(const vk::raii::Device& dev)
    : m_device(dev)
{
    createLayout();
    createPool();
    allocateSets();
}

vk::PipelineLayout DescriptorManager::pipelineLayout() const
    { return *m_pipeLayout; }

vk::DescriptorSet DescriptorManager::descriptorSet(uint32_t i) const
    { return *m_sets[i]; }

// ---- Layout ----
void DescriptorManager::createLayout() {
    using DS = vk::DescriptorSetLayoutBinding;
    using DT = vk::DescriptorType;
    using SS = vk::ShaderStageFlagBits;
    auto sb = [](uint32_t b, DT t) {
        return DS(b, t, 1, SS::eCompute); };
    std::vector<DS> b = {
        sb(0, DT::eAccelerationStructureKHR),
        sb(1, DT::eStorageImage),
        sb(2, DT::eUniformBuffer),
        sb(3, DT::eUniformBuffer),
        sb(4, DT::eStorageBuffer),
        sb(5, DT::eStorageBuffer),
        sb(6, DT::eStorageBuffer),
        sb(7, DT::eStorageBuffer),
        sb(8, DT::eStorageBuffer),
        sb(9, DT::eStorageBuffer),
        sb(10, DT::eStorageBuffer),
        sb(11, DT::eCombinedImageSampler),
        sb(12, DT::eStorageBuffer),
        sb(13, DT::eStorageBuffer),
        sb(14, DT::eStorageImage),
        sb(15, DT::eStorageImage),
        sb(16, DT::eStorageBuffer),
    };
    m_layout = vk::raii::DescriptorSetLayout(m_device, {{}, b});

    vk::PushConstantRange pc(
        vk::ShaderStageFlagBits::eCompute, 0, 112);
    m_pipeLayout = vk::raii::PipelineLayout(
        m_device, {{}, *m_layout, pc});
}

// ---- Pool ----
void DescriptorManager::createPool() {
    using DT = vk::DescriptorType;
    constexpr auto N = MAX_FRAMES_IN_FLIGHT;
    std::vector<vk::DescriptorPoolSize> sizes = {
        {DT::eAccelerationStructureKHR, N},
        {DT::eStorageImage,             N * 3},
        {DT::eUniformBuffer,            N * 2},
        {DT::eStorageBuffer,            N * 10},
        {DT::eCombinedImageSampler,     N},
    };
    m_pool = vk::raii::DescriptorPool(
        m_device, {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, N, sizes});
}

void DescriptorManager::allocateSets() {
    m_sets.reserve(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorSetAllocateInfo ai(*m_pool, *m_layout);
        auto sets = vk::raii::DescriptorSets(m_device, ai);
        m_sets.emplace_back(std::move(sets[0]));
    }
}

// ---- Bind helpers ----
static void writeBuf(const vk::raii::Device& d,
    vk::DescriptorSet ds, uint32_t b,
    vk::DescriptorType t, vk::Buffer buf, vk::DeviceSize sz)
{
    vk::DescriptorBufferInfo info(buf, 0, sz);
    d.updateDescriptorSets(
        vk::WriteDescriptorSet(ds, b, 0, 1, t, nullptr, &info), nullptr);
}

static void writeImg(const vk::raii::Device& d,
    vk::DescriptorSet ds, uint32_t b,
    vk::DescriptorType t, vk::ImageView v, vk::Sampler s)
{
    vk::DescriptorImageInfo info(s, v, vk::ImageLayout::eGeneral);
    d.updateDescriptorSets(
        vk::WriteDescriptorSet(ds, b, 0, 1, t, &info), nullptr);
}

void DescriptorManager::bindTLAS(uint32_t fi, vk::AccelerationStructureKHR tlas) {
    vk::WriteDescriptorSetAccelerationStructureKHR asInfo(1, &tlas);
    m_device.updateDescriptorSets(
        vk::WriteDescriptorSet(*m_sets[fi], 0, 0, 1,
            vk::DescriptorType::eAccelerationStructureKHR,
            nullptr, nullptr, nullptr, &asInfo), nullptr);
}

void DescriptorManager::bindOutputImage(uint32_t fi, vk::ImageView v, vk::Sampler s)
    { writeImg(m_device, *m_sets[fi], 1, vk::DescriptorType::eStorageImage, v, s); }

void DescriptorManager::bindMaterialBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 2, vk::DescriptorType::eUniformBuffer, b, sz); }

void DescriptorManager::bindLightBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 3, vk::DescriptorType::eUniformBuffer, b, sz); }

void DescriptorManager::bindGeometrySSBOs(uint32_t fi,
    vk::Buffer vBuf, vk::DeviceSize vSz,
    vk::Buffer iBuf, vk::DeviceSize iSz,
    vk::Buffer rBuf, vk::DeviceSize rSz)
{
    std::vector<vk::WriteDescriptorSet> w;
    vk::DescriptorBufferInfo vi(vBuf, 0, vSz);
    w.emplace_back(*m_sets[fi], 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &vi);
    vk::DescriptorBufferInfo ii(iBuf, 0, iSz);
    w.emplace_back(*m_sets[fi], 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &ii);
    vk::DescriptorBufferInfo ri(rBuf, 0, rSz);
    w.emplace_back(*m_sets[fi], 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &ri);
    m_device.updateDescriptorSets(w, nullptr);
}

void DescriptorManager::bindRayBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 7, vk::DescriptorType::eStorageBuffer, b, sz); }
void DescriptorManager::bindCounterBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 8, vk::DescriptorType::eStorageBuffer, b, sz); }
void DescriptorManager::bindPixelAccum(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 9, vk::DescriptorType::eStorageBuffer, b, sz); }
void DescriptorManager::bindOverflowBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 10, vk::DescriptorType::eStorageBuffer, b, sz); }

void DescriptorManager::bindEnvMap(uint32_t fi, vk::ImageView v, vk::Sampler s) {
    vk::DescriptorImageInfo info(s, v, vk::ImageLayout::eShaderReadOnlyOptimal);
    m_device.updateDescriptorSets(
        vk::WriteDescriptorSet(*m_sets[fi], 11, 0, 1,
            vk::DescriptorType::eCombinedImageSampler, &info), nullptr);
}

void DescriptorManager::bindNormalSSBO(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 12, vk::DescriptorType::eStorageBuffer, b, sz); }
void DescriptorManager::bindAccumBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 13, vk::DescriptorType::eStorageBuffer, b, sz); }

void DescriptorManager::bindNormalImage(uint32_t fi, vk::ImageView v)
    { writeImg(m_device, *m_sets[fi], 14, vk::DescriptorType::eStorageImage, v, {}); }
void DescriptorManager::bindDepthImage(uint32_t fi, vk::ImageView v)
    { writeImg(m_device, *m_sets[fi], 15, vk::DescriptorType::eStorageImage, v, {}); }
void DescriptorManager::bindInstanceNormalBuffer(uint32_t fi, vk::Buffer b, vk::DeviceSize sz)
    { writeBuf(m_device, *m_sets[fi], 16, vk::DescriptorType::eStorageBuffer, b, sz); }
