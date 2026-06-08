#pragma once

#include "GPUBuffer.h"
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <vector>

/// Manages BLAS (Bottom-Level) and TLAS (Top-Level) acceleration structures
/// for ray tracing. Uses VK_KHR_acceleration_structure.
class AccelerationStructure {
public:
    /// Input vertex data (positions only, interleaved as xyz floats).
    struct MeshData {
        std::vector<float>    vertices;   // interleaved xyz
        std::vector<uint32_t> indices;    // triangle indices
    };

    AccelerationStructure(
        const vk::raii::Device&         device,
        const vk::raii::PhysicalDevice& physDevice,
        uint32_t                        computeQueueFamily
    );
    ~AccelerationStructure();

    // Non-copyable, non-movable
    AccelerationStructure(const AccelerationStructure&)            = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure(AccelerationStructure&&)                 = delete;
    AccelerationStructure& operator=(AccelerationStructure&&)      = delete;

    /// Build BLAS from mesh data, then wrap it in a single-instance TLAS.
    void build(const MeshData& mesh);

    /// Get the TLAS handle for descriptor binding.
    vk::AccelerationStructureKHR getTLAS() const { return *m_tlas; }

    /// Device address of the TLAS (for shader binding).
    vk::DeviceAddress getTLASAddress() const { return m_tlasAddress; }

private:
    void createCommandPool(uint32_t computeQueueFamily);
    vk::raii::CommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(vk::raii::CommandBuffer& cmdBuf);

    void buildBLAS(const MeshData& mesh);
    void buildTLAS();

    const vk::raii::Device&          m_device;
    const vk::raii::PhysicalDevice&  m_physDevice;

    vk::raii::CommandPool            m_commandPool        = nullptr;

    // BLAS
    vk::raii::AccelerationStructureKHR m_blas            = nullptr;
    vk::DeviceAddress                   m_blasAddress     = 0;

    // TLAS
    vk::raii::AccelerationStructureKHR m_tlas             = nullptr;
    vk::DeviceAddress                   m_tlasAddress     = 0;

    // Temporary build buffers (freed after build completes)
    GPUBuffer m_scratchBuffer;
    GPUBuffer m_blasBuffer;
    GPUBuffer m_tlasBuffer;
    GPUBuffer m_instanceBuffer;
};
