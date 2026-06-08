#include "AccelerationStructure.h"

#include <cstring>
#include <stdexcept>

AccelerationStructure::AccelerationStructure(
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physDevice,
    uint32_t                        computeQueueFamily)
    : m_device(device)
    , m_physDevice(physDevice)
    , m_queueFamily(computeQueueFamily)
{
    createCommandPool(computeQueueFamily);
}

AccelerationStructure::~AccelerationStructure() {
    // vk::raii destructors handle cleanup via RAII.
    // Destruction order: TLAS → BLAS list → scratch → instance → material → command pool
}

// -----------------------------------------------------------------------------
// Command pool & helper buffers
// -----------------------------------------------------------------------------
void AccelerationStructure::createCommandPool(uint32_t queueFamily) {
    vk::CommandPoolCreateInfo poolInfo(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily
    );
    m_commandPool = vk::raii::CommandPool(m_device, poolInfo);
}

vk::raii::CommandBuffer AccelerationStructure::beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo(*m_commandPool, vk::CommandBufferLevel::ePrimary, 1);
    auto cmdBufs = vk::raii::CommandBuffers(m_device, allocInfo);
    auto cmdBuf  = std::move(cmdBufs[0]);

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuf.begin(beginInfo);
    return cmdBuf;
}

void AccelerationStructure::endSingleTimeCommands(vk::raii::CommandBuffer& cmdBuf) {
    cmdBuf.end();

    vk::SubmitInfo submitInfo({}, {}, *cmdBuf);
    auto queue = m_device.getQueue(m_queueFamily, 0);
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

// -----------------------------------------------------------------------------
// Build a single BLAS from mesh data (returns handle + address + buffer)
// -----------------------------------------------------------------------------
AccelerationStructure::BlasResult
AccelerationStructure::buildSingleBLAS(const MeshData& mesh) {
    auto cmdBuf = beginSingleTimeCommands();

    // --- Vertex buffer ---
    vk::DeviceSize vertSize = mesh.vertices.size() * sizeof(float);
    auto vertBuf = GPUBuffer::createStaging(
        m_device, mesh.vertices.data(), vertSize,
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        m_physDevice
    );

    // --- Index buffer ---
    vk::DeviceSize idxSize = mesh.indices.size() * sizeof(uint32_t);
    auto idxBuf = GPUBuffer::createStaging(
        m_device, mesh.indices.data(), idxSize,
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        m_physDevice
    );

    // --- Geometry description ---
    vk::AccelerationStructureGeometryTrianglesDataKHR triData(
        vk::Format::eR32G32B32Sfloat,
        vertBuf.address,
        sizeof(float) * 3,
        static_cast<uint32_t>(mesh.vertices.size() / 3) - 1,
        vk::IndexType::eUint32,
        idxBuf.address,
        {}
    );

    vk::AccelerationStructureGeometryKHR asGeom(
        vk::GeometryTypeKHR::eTriangles, triData,
        vk::GeometryFlagBitsKHR::eOpaque
    );

    uint32_t maxPrimitiveCount = static_cast<uint32_t>(mesh.indices.size()) / 3;

    vk::AccelerationStructureBuildGeometryInfoKHR buildGeomInfo(
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        vk::BuildAccelerationStructureModeKHR::eBuild
    );
    buildGeomInfo.setGeometries(asGeom);

    auto sizeInfo = m_device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        buildGeomInfo,
        maxPrimitiveCount
    );

    // --- Allocate BLAS buffer ---
    BlasResult result;
    result.buffer = GPUBuffer::create(
        m_device, sizeInfo.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );

    vk::AccelerationStructureCreateInfoKHR asCreateInfo(
        {}, *result.buffer.buffer, 0, sizeInfo.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eBottomLevel, 0
    );
    result.blas = vk::raii::AccelerationStructureKHR(m_device, asCreateInfo);
    result.address = m_device.getAccelerationStructureAddressKHR(
        vk::AccelerationStructureDeviceAddressInfoKHR(*result.blas)
    );

    buildGeomInfo.setDstAccelerationStructure(*result.blas);

    // --- Scratch buffer ---
    GPUBuffer scratchBuf = GPUBuffer::create(
        m_device, sizeInfo.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );
    buildGeomInfo.setScratchData(m_device.getBufferAddress(
        vk::BufferDeviceAddressInfo(*scratchBuf.buffer)
    ));

    // --- Build ---
    vk::AccelerationStructureBuildRangeInfoKHR range(maxPrimitiveCount, 0, 0, 0);
    vk::AccelerationStructureBuildRangeInfoKHR* pRange = &range;
    cmdBuf.buildAccelerationStructuresKHR({buildGeomInfo}, {pRange});

    endSingleTimeCommands(cmdBuf);

    return result;
}

// -----------------------------------------------------------------------------
// Upload material data to a host-visible uniform buffer
// -----------------------------------------------------------------------------
void AccelerationStructure::uploadMaterialBuffer(const std::vector<MaterialGPU>& data) {
    m_materialCount = static_cast<uint32_t>(data.size());
    vk::DeviceSize bufSize = MAX_MATERIALS * sizeof(MaterialGPU);

    m_materialBuffer = GPUBuffer::create(
        m_device, bufSize,
        vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );

    // Map and copy material data
    void* mapped = m_materialBuffer.memory.mapMemory(0, bufSize);
    std::memset(mapped, 0, static_cast<size_t>(bufSize));
    std::memcpy(mapped, data.data(),
                static_cast<size_t>(data.size() * sizeof(MaterialGPU)));
    m_materialBuffer.memory.unmapMemory();
}

// -----------------------------------------------------------------------------
// Build TLAS from N instances (BLAS must already be in m_blasList)
// -----------------------------------------------------------------------------
void AccelerationStructure::buildTLAS(uint32_t instanceCount) {
    auto cmdBuf = beginSingleTimeCommands();

    // --- Build instance descriptors ---
    std::vector<VkAccelerationStructureInstanceKHR> instances(instanceCount);
    for (uint32_t i = 0; i < instanceCount; ++i) {
        VkAccelerationStructureInstanceKHR& inst = instances[i];
        // Identity transform
        std::memset(&inst.transform, 0, sizeof(inst.transform));
        for (int r = 0; r < 3; ++r) {
            inst.transform.matrix[r][r] = 1.0f;
        }
        inst.instanceCustomIndex                    = i;   // used as face/material index in shader
        inst.mask                                   = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference         = m_blasList[i].address;
    }

    // --- Instance buffer ---
    vk::DeviceSize instanceBufSize = instanceCount * sizeof(VkAccelerationStructureInstanceKHR);
    m_instanceBuffer = GPUBuffer::createStaging(
        m_device, instances.data(), instanceBufSize,
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        m_physDevice
    );

    // --- Geometry ---
    vk::AccelerationStructureGeometryInstancesDataKHR instData(
        false, m_instanceBuffer.address
    );
    vk::AccelerationStructureGeometryKHR asGeom(
        vk::GeometryTypeKHR::eInstances, instData
    );

    // --- Build sizes ---
    vk::AccelerationStructureBuildGeometryInfoKHR buildGeomInfo(
        vk::AccelerationStructureTypeKHR::eTopLevel,
        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        vk::BuildAccelerationStructureModeKHR::eBuild
    );
    buildGeomInfo.setGeometries(asGeom);

    auto sizeInfo = m_device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        buildGeomInfo,
        instanceCount
    );

    // --- Allocate TLAS buffer ---
    m_tlasBuffer = GPUBuffer::create(
        m_device, sizeInfo.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );

    vk::AccelerationStructureCreateInfoKHR asCreateInfo(
        {}, *m_tlasBuffer.buffer, 0, sizeInfo.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eTopLevel, 0
    );
    m_tlas = vk::raii::AccelerationStructureKHR(m_device, asCreateInfo);
    m_tlasAddress = m_device.getAccelerationStructureAddressKHR(
        vk::AccelerationStructureDeviceAddressInfoKHR(*m_tlas)
    );
    buildGeomInfo.setDstAccelerationStructure(*m_tlas);

    // --- Scratch buffer ---
    m_scratchBuffer = GPUBuffer::create(
        m_device, sizeInfo.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );
    buildGeomInfo.setScratchData(m_device.getBufferAddress(
        vk::BufferDeviceAddressInfo(*m_scratchBuffer.buffer)
    ));

    // --- Build ---
    vk::AccelerationStructureBuildRangeInfoKHR range(instanceCount, 0, 0, 0);
    vk::AccelerationStructureBuildRangeInfoKHR* pRange = &range;
    cmdBuf.buildAccelerationStructuresKHR({buildGeomInfo}, {pRange});

    endSingleTimeCommands(cmdBuf);
}

// -----------------------------------------------------------------------------
// Phase 1: single mesh → single BLAS → single-instance TLAS
// -----------------------------------------------------------------------------
void AccelerationStructure::build(const MeshData& mesh) {
    m_blasList.clear();
    m_blasList.push_back(buildSingleBLAS(mesh));
    buildTLAS(1);
}

// -----------------------------------------------------------------------------
// Phase 2: two meshes → two BLAS → two-instance TLAS + material UBO
// -----------------------------------------------------------------------------
void AccelerationStructure::buildTwoInstance(
    const InstanceInfo& inst0,
    const InstanceInfo& inst1,
    const std::vector<MaterialGPU>& materialData)
{
    m_blasList.clear();
    m_blasList.push_back(buildSingleBLAS(inst0.mesh));
    m_blasList.push_back(buildSingleBLAS(inst1.mesh));

    // Override the instanceCustomIndex to match what the shader expects
    // (we'll set it in buildTLAS based on the order in the instances array)
    buildTLAS(2);

    uploadMaterialBuffer(materialData);
}
