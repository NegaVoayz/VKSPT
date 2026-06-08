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
    // vk::raii destructors handle cleanup via RAII
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
    // Use the correct queue family for AS builds
    auto queue = m_device.getQueue(m_queueFamily, 0);
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

// -----------------------------------------------------------------------------
// Main build entry point
// -----------------------------------------------------------------------------
void AccelerationStructure::build(const MeshData& mesh) {
    buildBLAS(mesh);
    buildTLAS();
}

// -----------------------------------------------------------------------------
// BLAS: single triangle mesh
// -----------------------------------------------------------------------------
void AccelerationStructure::buildBLAS(const MeshData& mesh) {
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
        vk::Format::eR32G32B32Sfloat,               // vertex format
        vertBuf.address,                              // vertex data address
        sizeof(float) * 3,                            // stride
        static_cast<uint32_t>(mesh.vertices.size() / 3) - 1, // max vertex
        vk::IndexType::eUint32,                       // index type
        idxBuf.address,                               // index data address
        {}                                            // transform (none)
    );

    vk::AccelerationStructureGeometryKHR asGeom(
        vk::GeometryTypeKHR::eTriangles, triData,
        vk::GeometryFlagBitsKHR::eOpaque  // no alpha testing in Phase 1
    );

    // --- Get build sizes ---
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
    m_blasBuffer = GPUBuffer::create(
        m_device, sizeInfo.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );

    // Create BLAS
    vk::AccelerationStructureCreateInfoKHR asCreateInfo(
        {}, *m_blasBuffer.buffer, 0, sizeInfo.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eBottomLevel, 0
    );
    m_blas = vk::raii::AccelerationStructureKHR(m_device, asCreateInfo);
    m_blasAddress = m_device.getAccelerationStructureAddressKHR(
        vk::AccelerationStructureDeviceAddressInfoKHR(*m_blas)
    );

    buildGeomInfo.setDstAccelerationStructure(*m_blas);

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
    vk::AccelerationStructureBuildRangeInfoKHR range(
        maxPrimitiveCount, 0, 0, 0
    );
    vk::AccelerationStructureBuildRangeInfoKHR* pRange = &range;
    cmdBuf.buildAccelerationStructuresKHR({buildGeomInfo}, {pRange});

    endSingleTimeCommands(cmdBuf);
}

// -----------------------------------------------------------------------------
// TLAS: single instance wrapping the BLAS
// -----------------------------------------------------------------------------
void AccelerationStructure::buildTLAS() {
    auto cmdBuf = beginSingleTimeCommands();

    // --- Instance ---
    VkAccelerationStructureInstanceKHR instance{};
    // Identity transform
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            instance.transform.matrix[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    instance.instanceCustomIndex                    = 0;
    instance.mask                                   = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference         = m_blasAddress;

    // Instance buffer (GPU-only, then upload)
    m_instanceBuffer = GPUBuffer::createStaging(
        m_device, &instance, sizeof(instance),
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
        1  // 1 instance
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

    // --- Scratch buffer (reuse BLAS scratch if large enough) ---
    m_scratchBuffer = GPUBuffer::create(
        m_device,
        std::max(sizeInfo.buildScratchSize, m_scratchBuffer.size),
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );
    buildGeomInfo.setScratchData(m_device.getBufferAddress(
        vk::BufferDeviceAddressInfo(*m_scratchBuffer.buffer)
    ));

    // --- Build ---
    vk::AccelerationStructureBuildRangeInfoKHR range(1, 0, 0, 0);
    vk::AccelerationStructureBuildRangeInfoKHR* pRange = &range;
    cmdBuf.buildAccelerationStructuresKHR({buildGeomInfo}, {pRange});

    endSingleTimeCommands(cmdBuf);
}
