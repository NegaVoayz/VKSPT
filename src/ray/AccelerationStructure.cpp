#include "ray/AccelerationStructure.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>
#include <stdexcept>

AccelerationStructure::AccelerationStructure(
    const vk::raii::Device& dev, const vk::raii::PhysicalDevice& pd,
    uint32_t qf)
    : m_device(dev), m_physDevice(pd), m_qf(qf), m_builder(dev, pd, qf) {}

AccelerationStructure::~AccelerationStructure() = default;

void AccelerationStructure::uploadMaterialBuffer(
    const std::vector<MaterialGPU>& data)
{
    m_matCount = uint32_t(data.size());
    vk::DeviceSize sz = maxMaterials * sizeof(MaterialGPU);
    m_matBuf = GPUBuffer::Create(m_device, sz,
        vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, m_physDevice);
    void* m = m_matBuf.Memory.mapMemory(0, sz);
    std::memset(m, 0, size_t(sz));
    std::memcpy(m, data.data(), data.size() * sizeof(MaterialGPU));
    m_matBuf.Memory.unmapMemory();
}

void AccelerationStructure::uploadLightBuffer(
    const std::vector<GpuLight>& lights)
{
    vk::DeviceSize sz = maxLights * sizeof(GpuLight);
    m_lightBuf = GPUBuffer::Create(m_device, sz,
        vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, m_physDevice);
    void* m = m_lightBuf.Memory.mapMemory(0, sz);
    std::memset(m, 0, size_t(sz));
    size_t n = std::min(lights.size(), size_t(maxLights));
    m_lightCount = uint32_t(n);
    m_lightsCPU.assign(lights.begin(), lights.begin() + n);
    std::memcpy(m, lights.data(), n * sizeof(GpuLight));
    m_lightBuf.Memory.unmapMemory();
}

void AccelerationStructure::build(const MeshData& mesh)
{
    m_instCount = 1;
    m_stagedV = {mesh.vertices};
    m_stagedI = {mesh.indices};
    m_stagedN = {mesh.normals};
    m_blasList.clear();
    m_blasList.push_back(m_builder.buildBLAS(mesh.vertices, mesh.indices));
    m_xfs = {{{ {1,0,0,0},{0,1,0,0},{0,0,1,0} }}};
    m_builder.buildTLAS(1, m_xfs, m_blasList,
        m_tlasBuf, m_scratch, m_instBuf, m_tlas, m_tlasAddr);
    m_geom.upload(m_device, m_physDevice, m_stagedV, m_stagedI, m_stagedN,
                  {0}, {0});
}

void AccelerationStructure::buildTwoInstance(
    const InstanceInfo& i0, const InstanceInfo& i1,
    const std::vector<MaterialGPU>& mats, const GpuLight& sky)
{ buildScene({i0, i1}, mats, {sky}); }

void AccelerationStructure::buildScene(
    const std::vector<InstanceInfo>& instances,
    const std::vector<MaterialGPU>& mats,
    const std::vector<GpuLight>&    lights)
{
    m_instCount = uint32_t(instances.size());
    m_stagedV.clear(); m_stagedI.clear(); m_stagedN.clear();
    m_blasList.clear(); m_xfs.clear();

    for (auto& inst : instances) {
        m_stagedV.push_back(inst.mesh.vertices);
        m_stagedI.push_back(inst.mesh.indices);
        m_stagedN.push_back(inst.mesh.normals);
        m_blasList.push_back(
            m_builder.buildBLAS(inst.mesh.vertices, inst.mesh.indices));
        m_xfs.push_back({{
            {inst.transform[0][0],inst.transform[0][1],inst.transform[0][2],inst.transform[0][3]},
            {inst.transform[1][0],inst.transform[1][1],inst.transform[1][2],inst.transform[1][3]},
            {inst.transform[2][0],inst.transform[2][1],inst.transform[2][2],inst.transform[2][3]}
        }});
    }
    m_builder.buildTLAS(m_instCount, m_xfs, m_blasList,
        m_tlasBuf, m_scratch, m_instBuf, m_tlas, m_tlasAddr);

    // Normal matrices
    {
        std::vector<float> nd(maxInstances * 3 * 4, 0);
        for (uint32_t i = 0; i < m_instCount; ++i) {
            glm::mat3 m33;
            for (int c=0;c<3;++c) for (int r=0;r<3;++r) m33[c][r]=m_xfs[i][r][c];
            glm::mat3 nm = glm::transpose(glm::inverse(m33));
            for (int c=0;c<3;++c) {
                size_t b = (c*maxInstances+i)*4;
                nd[b]=nm[c].x; nd[b+1]=nm[c].y; nd[b+2]=nm[c].z;
            }
        }
        m_normBuf = GPUBuffer::CreateStaging(m_device, nd.data(),
            nd.size()*sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer, m_physDevice);
    }

    std::vector<uint32_t> mids, sfs;
    for (size_t i=0;i<instances.size();++i) {
        mids.push_back(instances[i].materialID);
        sfs.push_back((instances[i].hasNormals && i<m_stagedN.size()
                       && !m_stagedN[i].empty()) ? 1u : 0u);
    }
    m_geom.maxInstances = maxInstances;
    m_geom.upload(m_device, m_physDevice, m_stagedV, m_stagedI, m_stagedN,
                  mids, sfs);
    uploadMaterialBuffer(mats);
    uploadLightBuffer(lights);
    createPhotonBuffers();
    createHashBuffers();
    createRayStatsBuffer();
}

void AccelerationStructure::createPhotonBuffers()
{
    constexpr uint32_t kPhotonRecSize = 88;  // PhotonRecord: 3×float4 + 2×float4 + float2

    m_photonBuf = GPUBuffer::Create(m_device,
        MAX_PHOTONS * kPhotonRecSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);

    m_photonCtr = GPUBuffer::Create(m_device, 4,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);
}

void AccelerationStructure::createHashBuffers()
{
    m_hashCellData = GPUBuffer::Create(m_device,
        HASH_TABLE_SIZE * 2 * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);

    m_sortedPhotonIndices = GPUBuffer::Create(m_device,
        MAX_PHOTONS * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);

    m_cellPhotonData = GPUBuffer::Create(m_device,
        HASH_TABLE_SIZE * 23 * sizeof(float),  // 23 floats per hash cell
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);

    m_gatheredCellData = GPUBuffer::Create(m_device,
        HASH_TABLE_SIZE * 10 * sizeof(float) + 12,  // +12B debug: 3 uint tail for overflow flag/mask/value
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);
}

void AccelerationStructure::createRayStatsBuffer()
{
    m_rayStats = GPUBuffer::Create(m_device, 1024,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_physDevice);
}
