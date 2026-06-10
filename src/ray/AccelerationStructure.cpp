#include "ray/AccelerationStructure.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstring>
#include <iostream>
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
    // Save vertex/index/normal data for later SSBO upload (shader normal computation)
    m_stagedVertices.push_back(mesh.vertices);
    m_stagedIndices.push_back(mesh.indices);
    m_stagedNormals.push_back(mesh.normals);

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
// Upload sky/environment light SPD (D65 illuminant)
// -----------------------------------------------------------------------------
void AccelerationStructure::uploadLightBuffer(const std::vector<GpuLight>& lights) {
    vk::DeviceSize bufSize = MAX_LIGHTS * sizeof(GpuLight);

    m_lightBuffer = GPUBuffer::create(
        m_device, bufSize,
        vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );

    // Upload up to MAX_LIGHTS elements
    void* mapped = m_lightBuffer.memory.mapMemory(0, bufSize);
    std::memset(mapped, 0, static_cast<size_t>(bufSize));
    size_t copyCount = std::min(lights.size(), size_t(MAX_LIGHTS));
    std::memcpy(mapped, lights.data(), copyCount * sizeof(GpuLight));
    m_lightBuffer.memory.unmapMemory();
}

// -----------------------------------------------------------------------------
void AccelerationStructure::buildTLAS(uint32_t instanceCount) {
    auto cmdBuf = beginSingleTimeCommands();

    // --- Build instance descriptors ---
    std::vector<VkAccelerationStructureInstanceKHR> instances(instanceCount);
    for (uint32_t i = 0; i < instanceCount; ++i) {
        VkAccelerationStructureInstanceKHR& inst = instances[i];
        // Use the per-instance 3×4 row-major transform
        if (i < m_instanceTransforms.size()) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    inst.transform.matrix[r][c] = m_instanceTransforms[i][r][c];
        } else {
            // Identity fallback
            std::memset(&inst.transform, 0, sizeof(inst.transform));
            for (int r = 0; r < 3; ++r)
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
    m_instanceCount = 1;
    m_stagedVertices.clear();
    m_stagedIndices.clear();

    m_blasList.clear();
    m_blasList.push_back(buildSingleBLAS(mesh));
    buildTLAS(1);
    std::vector<uint32_t> mid = {0}, sfl = {0};
    m_geometry.upload(m_device, m_physDevice,
        m_stagedVertices, m_stagedIndices, m_stagedNormals, mid, sfl);
}

// -----------------------------------------------------------------------------
// Phase 2: two meshes → two BLAS → two-instance TLAS + material UBO
// -----------------------------------------------------------------------------
void AccelerationStructure::buildTwoInstance(
    const InstanceInfo& inst0,
    const InstanceInfo& inst1,
    const std::vector<MaterialGPU>& materialData,
    const GpuLight& skyLight)
{
    buildScene({inst0, inst1}, materialData, {skyLight});
}

void AccelerationStructure::buildScene(
    const std::vector<InstanceInfo>& instances,
    const std::vector<MaterialGPU>& materialData,
    const std::vector<GpuLight>&    lights)
{
    m_instanceCount = static_cast<uint32_t>(instances.size());
    m_stagedVertices.clear();
    m_stagedIndices.clear();
    m_stagedNormals.clear();
    m_blasList.clear();
    m_instanceTransforms.clear();

    for (const auto& inst : instances) {
        m_blasList.push_back(buildSingleBLAS(inst.mesh));
        m_instanceTransforms.push_back(
            std::array<std::array<float,4>,3>{{
                {inst.transform[0][0], inst.transform[0][1], inst.transform[0][2], inst.transform[0][3]},
                {inst.transform[1][0], inst.transform[1][1], inst.transform[1][2], inst.transform[1][3]},
                {inst.transform[2][0], inst.transform[2][1], inst.transform[2][2], inst.transform[2][3]}
            }}
        );
    }

    buildTLAS(m_instanceCount);

    // ---- Build per-instance normal matrices (inverse-transpose of 3×3 affine) ----
    // The normal matrix transforms object-space normals to world-space.
    // Stored as 3 arrays of vec4 (columns 0,1,2), MAX_INSTANCES each, std430.
    {
        std::vector<float> normalData(MAX_INSTANCES * 3 * 4, 0.0f);
        for (uint32_t inst = 0; inst < m_instanceCount; ++inst) {
            // Convert row-major 3×4 to column-major glm::mat3
            const auto& xf = m_instanceTransforms[inst];
            glm::mat3 m33;
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r)
                    m33[c][r] = xf[r][c];  // row-major xf → column-major glm

            glm::mat3 normalMat = glm::transpose(glm::inverse(m33));

            // Store columns: col0 = normalMat[0], col1 = normalMat[1], col2 = normalMat[2]
            for (int col = 0; col < 3; ++col) {
                size_t base = (col * MAX_INSTANCES + inst) * 4;
                normalData[base + 0] = normalMat[col].x;
                normalData[base + 1] = normalMat[col].y;
                normalData[base + 2] = normalMat[col].z;
                normalData[base + 3] = 0.0f;  // padding
            }
        }

        m_instanceNormalBuffer = GPUBuffer::createStaging(
            m_device, normalData.data(), normalData.size() * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer, m_physDevice);
    }

    // Collect per-instance metadata
    std::vector<uint32_t> matIDs, smoothFlags;
    for (size_t i = 0; i < instances.size(); ++i) {
        matIDs.push_back(instances[i].materialID);
        bool hasN = instances[i].hasNormals &&
                    i < m_stagedNormals.size() &&
                    !m_stagedNormals[i].empty();
        smoothFlags.push_back(hasN ? 1u : 0u);
    }

    m_geometry.upload(m_device, m_physDevice,
        m_stagedVertices, m_stagedIndices, m_stagedNormals,
        matIDs, smoothFlags);

    uploadMaterialBuffer(materialData);
    uploadLightBuffer(lights);
}

// loadEnvMap is now a thin wrapper in the header, delegating to EnvMap.
