#include "ray/AccelBuilder.h"
#include <cstring>

AccelBuilder::AccelBuilder(const vk::raii::Device&         dev,
                           const vk::raii::PhysicalDevice& pdev,
                           uint32_t                        qf)
    : m_dev(dev), m_pdev(pdev), m_qf(qf),
      m_pool(dev, {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, qf}) {}

vk::raii::CommandBuffer AccelBuilder::begin() {
    auto cbs = vk::raii::CommandBuffers(m_dev,
        {*m_pool, vk::CommandBufferLevel::ePrimary, 1});
    auto cb = std::move(cbs[0]);
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return cb;
}

void AccelBuilder::end(vk::raii::CommandBuffer& cb) {
    cb.end();
    vk::SubmitInfo si({}, {}, *cb);
    m_dev.getQueue(m_qf, 0).submit(si, nullptr);
    m_dev.waitIdle();
}

AccelBuilder::BlasResult AccelBuilder::buildBLAS(
    const std::vector<float>&    verts,
    const std::vector<uint32_t>& inds)
{
    auto cb = begin();

    auto vBuf = GPUBuffer::CreateStaging(m_dev, verts.data(),
        verts.size()*sizeof(float),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress, m_pdev);
    auto iBuf = GPUBuffer::CreateStaging(m_dev, inds.data(),
        inds.size()*sizeof(uint32_t),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress, m_pdev);

    vk::AccelerationStructureGeometryTrianglesDataKHR td(
        vk::Format::eR32G32B32Sfloat, vBuf.Address,
        sizeof(float)*3, uint32_t(verts.size()/3)-1,
        vk::IndexType::eUint32, iBuf.Address, {});
    vk::AccelerationStructureGeometryKHR g(
        vk::GeometryTypeKHR::eTriangles, td,
        vk::GeometryFlagBitsKHR::eOpaque);

    uint32_t n = uint32_t(inds.size())/3;
    vk::AccelerationStructureBuildGeometryInfoKHR bi(
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        vk::BuildAccelerationStructureModeKHR::eBuild);
    bi.setGeometries(g);

    auto sz = m_dev.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, bi, n);

    BlasResult r;
    r.buf = GPUBuffer::Create(m_dev, sz.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_pdev);
    r.blas = vk::raii::AccelerationStructureKHR(m_dev,
        {{}, *r.buf.Buffer, 0, sz.accelerationStructureSize,
         vk::AccelerationStructureTypeKHR::eBottomLevel, 0});
    r.addr = m_dev.getAccelerationStructureAddressKHR(
        {*r.blas});
    bi.setDstAccelerationStructure(*r.blas);

    auto scratch = GPUBuffer::Create(m_dev, sz.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_pdev);
    bi.setScratchData(m_dev.getBufferAddress({*scratch.Buffer}));

    vk::AccelerationStructureBuildRangeInfoKHR rng(n,0,0,0);
    auto* pr = &rng;
    cb.buildAccelerationStructuresKHR({bi}, {pr});
    end(cb);
    return r;
}

void AccelBuilder::buildTLAS(uint32_t n,
    const std::vector<std::array<std::array<float,4>,3>>& xfs,
    const std::vector<BlasResult>& blases,
    GPUBuffer& tBuf, GPUBuffer& sBuf, GPUBuffer& iBuf,
    vk::raii::AccelerationStructureKHR& tlas,
    vk::DeviceAddress& tAddr)
{
    auto cb = begin();

    std::vector<VkAccelerationStructureInstanceKHR> insts(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto& ii = insts[i];
        if (i < xfs.size())
            for (int r=0;r<3;++r) for (int c=0;c<4;++c)
                ii.transform.matrix[r][c] = xfs[i][r][c];
        else {
            std::memset(&ii.transform,0,sizeof(ii.transform));
            for (int r=0;r<3;++r) ii.transform.matrix[r][r]=1;
        }
        ii.instanceCustomIndex = i;
        ii.mask = 0xFF;
        ii.instanceShaderBindingTableRecordOffset = 0;
        ii.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        ii.accelerationStructureReference = blases[i].addr;
    }

    iBuf = GPUBuffer::CreateStaging(m_dev, insts.data(),
        n*sizeof(VkAccelerationStructureInstanceKHR),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress, m_pdev);

    vk::AccelerationStructureGeometryInstancesDataKHR id(false, iBuf.Address);
    vk::AccelerationStructureGeometryKHR g(
        vk::GeometryTypeKHR::eInstances, id);

    vk::AccelerationStructureBuildGeometryInfoKHR bi(
        vk::AccelerationStructureTypeKHR::eTopLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        vk::BuildAccelerationStructureModeKHR::eBuild);
    bi.setGeometries(g);

    auto sz = m_dev.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, bi, n);

    tBuf = GPUBuffer::Create(m_dev, sz.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_pdev);
    tlas = vk::raii::AccelerationStructureKHR(m_dev,
        {{}, *tBuf.Buffer, 0, sz.accelerationStructureSize,
         vk::AccelerationStructureTypeKHR::eTopLevel, 0});
    tAddr = m_dev.getAccelerationStructureAddressKHR({*tlas});
    bi.setDstAccelerationStructure(*tlas);

    sBuf = GPUBuffer::Create(m_dev, sz.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal, m_pdev);
    bi.setScratchData(m_dev.getBufferAddress({*sBuf.Buffer}));

    vk::AccelerationStructureBuildRangeInfoKHR rng(n,0,0,0);
    auto* pr = &rng;
    cb.buildAccelerationStructuresKHR({bi}, {pr});
    end(cb);
}
