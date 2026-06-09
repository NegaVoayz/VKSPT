#include "AccelerationStructure.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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
void AccelerationStructure::uploadLightBuffer(const GpuLight& skyLight) {
    vk::DeviceSize bufSize = MAX_LIGHTS * sizeof(GpuLight);

    m_lightBuffer = GPUBuffer::create(
        m_device, bufSize,
        vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );

    // Upload MAX_LIGHTS elements (first = sky light, rest = zeroed)
    void* mapped = m_lightBuffer.memory.mapMemory(0, bufSize);
    std::memset(mapped, 0, static_cast<size_t>(bufSize));
    std::memcpy(mapped, &skyLight, sizeof(GpuLight));
    m_lightBuffer.memory.unmapMemory();
}

// -----------------------------------------------------------------------------
// Upload vertex/index data as SSBOs for shader geometric normal computation
// -----------------------------------------------------------------------------
void AccelerationStructure::uploadVertexSSBO() {
    // Build SoA (struct-of-arrays) layout matching shader's InstanceRangeBlock:
    //   6 arrays × MAX_INSTANCES uints (now includes useSmoothNormals).
    std::vector<uint32_t> rangeData(6 * MAX_INSTANCES, 0);
    auto* vtxOff    = &rangeData[0 * MAX_INSTANCES];
    auto* vtxCnt    = &rangeData[1 * MAX_INSTANCES];
    auto* idxOff    = &rangeData[2 * MAX_INSTANCES];
    auto* idxCnt    = &rangeData[3 * MAX_INSTANCES];
    auto* matIDs    = &rangeData[4 * MAX_INSTANCES];
    auto* smoothN   = &rangeData[5 * MAX_INSTANCES];

    // Concatenate all vertex and index data across instances
    std::vector<float>    allVertices;
    std::vector<uint32_t> allIndices;
    std::vector<float>    allNormals;

    for (size_t inst = 0; inst < m_stagedVertices.size() && inst < MAX_INSTANCES; ++inst) {
        vtxOff[inst] = static_cast<uint32_t>(allVertices.size());
        vtxCnt[inst] = static_cast<uint32_t>(m_stagedVertices[inst].size());
        idxOff[inst] = static_cast<uint32_t>(allIndices.size());
        idxCnt[inst] = static_cast<uint32_t>(m_stagedIndices[inst].size());
        matIDs[inst] = 0;  // default material; caller can update for specific instances
        smoothN[inst] = 0;  // Phase 1 path: no smooth normals

        allVertices.insert(allVertices.end(),
            m_stagedVertices[inst].begin(), m_stagedVertices[inst].end());
        allIndices.insert(allIndices.end(),
            m_stagedIndices[inst].begin(), m_stagedIndices[inst].end());
        if (inst < m_stagedNormals.size() && !m_stagedNormals[inst].empty()) {
            allNormals.insert(allNormals.end(),
                m_stagedNormals[inst].begin(), m_stagedNormals[inst].end());
        }
    }

    // Upload vertex data SSBO
    if (!allVertices.empty()) {
        vk::DeviceSize vSize = allVertices.size() * sizeof(float);
        m_vertexDataBuffer = GPUBuffer::createStaging(
            m_device, allVertices.data(), vSize,
            vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice
        );
    }

    // Upload index data SSBO
    if (!allIndices.empty()) {
        vk::DeviceSize iSize = allIndices.size() * sizeof(uint32_t);
        m_indexDataBuffer = GPUBuffer::createStaging(
            m_device, allIndices.data(), iSize,
            vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice
        );
    }

    // Upload normal data SSBO
    if (!allNormals.empty()) {
        m_normalDataBuffer = GPUBuffer::createStaging(m_device, allNormals.data(),
            allNormals.size() * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice);
    } else {
        float dummy[4] = {0, 1, 0, 0};
        m_normalDataBuffer = GPUBuffer::createStaging(m_device, dummy, sizeof(dummy),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice);
    }

    // Upload instance range SSBO
    vk::DeviceSize rangeSize = rangeData.size() * sizeof(uint32_t);
    m_rangeBuffer = GPUBuffer::createStaging(
        m_device, rangeData.data(), rangeSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eShaderDeviceAddress,
        m_physDevice
    );
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
    uploadVertexSSBO();
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
    buildScene({inst0, inst1}, materialData, skyLight);
}

void AccelerationStructure::buildScene(
    const std::vector<InstanceInfo>& instances,
    const std::vector<MaterialGPU>& materialData,
    const GpuLight& skyLight)
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

    // Build range data with correct material IDs (6 arrays now: +useSmoothNormals)
    std::vector<uint32_t> rangeData(6 * MAX_INSTANCES, 0);
    auto* vtxOff  = &rangeData[0 * MAX_INSTANCES];
    auto* vtxCnt  = &rangeData[1 * MAX_INSTANCES];
    auto* idxOff  = &rangeData[2 * MAX_INSTANCES];
    auto* idxCnt  = &rangeData[3 * MAX_INSTANCES];
    auto* matIDs  = &rangeData[4 * MAX_INSTANCES];
    auto* smoothN = &rangeData[5 * MAX_INSTANCES];

    std::vector<float>    allVertices;
    std::vector<uint32_t> allIndices;
    std::vector<float>    allNormals;

    for (size_t inst = 0; inst < m_stagedVertices.size(); ++inst) {
        vtxOff[inst] = static_cast<uint32_t>(allVertices.size());
        vtxCnt[inst] = static_cast<uint32_t>(m_stagedVertices[inst].size());
        idxOff[inst] = static_cast<uint32_t>(allIndices.size());
        idxCnt[inst] = static_cast<uint32_t>(m_stagedIndices[inst].size());
        matIDs[inst] = instances[inst].materialID;
        smoothN[inst] = (instances[inst].hasNormals &&
                         !m_stagedNormals.empty() &&
                         !m_stagedNormals[inst].empty()) ? 1u : 0u;

        allVertices.insert(allVertices.end(),
            m_stagedVertices[inst].begin(), m_stagedVertices[inst].end());
        allIndices.insert(allIndices.end(),
            m_stagedIndices[inst].begin(), m_stagedIndices[inst].end());
        if (inst < m_stagedNormals.size()) {
            allNormals.insert(allNormals.end(),
                m_stagedNormals[inst].begin(), m_stagedNormals[inst].end());
        }
    }

    // Upload vertex SSBO
    if (!allVertices.empty()) {
        m_vertexDataBuffer = GPUBuffer::createStaging(m_device, allVertices.data(),
            allVertices.size() * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice);
    }
    // Upload index SSBO
    if (!allIndices.empty()) {
        m_indexDataBuffer = GPUBuffer::createStaging(m_device, allIndices.data(),
            allIndices.size() * sizeof(uint32_t),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice);
    }
    // Upload normal SSBO (same indexing as vertex SSBO)
    // Always create a valid buffer to satisfy binding 12 (Vulkan requires non-null)
    if (!allNormals.empty()) {
        m_normalDataBuffer = GPUBuffer::createStaging(m_device, allNormals.data(),
            allNormals.size() * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice);
    } else {
        // Dummy 4-float buffer so the binding is never null
        float dummy[4] = {0, 1, 0, 0};
        m_normalDataBuffer = GPUBuffer::createStaging(m_device, dummy, sizeof(dummy),
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            m_physDevice);
    }
    // Upload range SSBO
    m_rangeBuffer = GPUBuffer::createStaging(m_device, rangeData.data(),
        rangeData.size() * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        m_physDevice);

    uploadMaterialBuffer(materialData);
    uploadLightBuffer(skyLight);
}

// -----------------------------------------------------------------------------
// Environment map loading
// -----------------------------------------------------------------------------
void AccelerationStructure::loadEnvMap(const std::string& path) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (!pixels) {
        throw std::runtime_error("Failed to load env map: " + path);
    }

    vk::DeviceSize imgSize = static_cast<vk::DeviceSize>(width) * height * 4;

    // Staging buffer for pixel data
    auto stagingBuf = GPUBuffer::createStaging(
        m_device, pixels, imgSize,
        vk::BufferUsageFlagBits::eTransferSrc, m_physDevice
    );
    stbi_image_free(pixels);

    // Create image
    vk::ImageCreateInfo imgInfo(
        {}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Srgb,
        vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
        1, 1, vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::SharingMode::eExclusive
    );
    m_envMapImage = vk::raii::Image(m_device, imgInfo);

    auto memReqs = m_envMapImage.getMemoryRequirements();
    uint32_t memTypeIdx = 0;
    auto memProps = m_physDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            memTypeIdx = i;
            break;
        }
    }
    vk::MemoryAllocateInfo memInfo(memReqs.size, memTypeIdx);
    m_envMapMemory = vk::raii::DeviceMemory(m_device, memInfo);
    m_envMapImage.bindMemory(*m_envMapMemory, 0);

    // Copy staging → image with layout transitions
    {
        auto cmdBuf = beginSingleTimeCommands();

        // Transition: undefined → transfer_dst
        vk::ImageMemoryBarrier barrier1(
            {}, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *m_envMapImage,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
        );
        cmdBuf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, barrier1
        );

        vk::BufferImageCopy copyRgn(
            0, 0, 0,
            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
            {0, 0, 0},
            {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}
        );
        cmdBuf.copyBufferToImage(
            *stagingBuf.buffer, *m_envMapImage,
            vk::ImageLayout::eTransferDstOptimal, copyRgn
        );

        // Transition: transfer_dst → shader_read_only
        vk::ImageMemoryBarrier barrier2(
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *m_envMapImage,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
        );
        cmdBuf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {}, {}, barrier2
        );

        endSingleTimeCommands(cmdBuf);
    }

    // Image view
    vk::ImageViewCreateInfo viewInfo(
        {}, *m_envMapImage, vk::ImageViewType::e2D,
        vk::Format::eR8G8B8A8Srgb,
        vk::ComponentMapping{},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );
    m_envMapView = vk::raii::ImageView(m_device, viewInfo);

    // Sampler
    vk::SamplerCreateInfo samplerInfo(
        {}, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        0.0f, false, 1.0f, false, vk::CompareOp::eNever,
        0.0f, 0.0f, vk::BorderColor::eFloatOpaqueBlack
    );
    m_envMapSampler = vk::raii::Sampler(m_device, samplerInfo);

    std::cout << "  Env map loaded: " << path << " (" << width << "x" << height << ")" << std::endl;
}
