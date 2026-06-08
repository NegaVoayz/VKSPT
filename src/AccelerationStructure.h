#pragma once

#include "GPUBuffer.h"
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <vector>

/// Manages BLAS (Bottom-Level) and TLAS (Top-Level) acceleration structures
/// for ray tracing. Uses VK_KHR_acceleration_structure.
/// Also owns the material uniform buffer uploaded to the GPU.
class AccelerationStructure {
public:
    /// Maximum number of materials in the UBO (matches shader array size).
    static constexpr uint32_t MAX_MATERIALS = 8;

    /// Input vertex data (positions only, interleaved as xyz floats).
    struct MeshData {
        std::vector<float>    vertices;   // interleaved xyz
        std::vector<uint32_t> indices;    // triangle indices
    };

    /// Descriptor for a single TLAS instance.
    struct InstanceInfo {
        MeshData mesh;
        uint32_t customIndex  = 0;   // rayQueryGetIntersectionInstanceCustomIndexEXT
        uint32_t materialID   = 0;   // index into material UBO
    };

    /// CPU-side material data uploaded to the GPU UBO.
    /// Must match GpuMaterial in spectral_common.slang (std140, 64 bytes).
    struct alignas(16) MaterialGPU {
        float cauchyA[4] = {};    // (R, G, B, _)
        float cauchyB[4] = {};    // (R, G, B, _) — B for λ in μm
        float albedo[4]   = {};   // (R, G, B, _)
        float params[4]   = {};   // (ior, roughness, type, _)
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

    /// Build BLAS from a single mesh, wrap in single-instance TLAS (Phase 1 compat).
    void build(const MeshData& mesh);

    /// Build two BLAS, then a TLAS with two instances (Phase 2 prism).
    /// Each instance gets its own customIndex (for shader identification)
    /// and materialID (index into the material UBO).
    void buildTwoInstance(
        const InstanceInfo& inst0,
        const InstanceInfo& inst1,
        const std::vector<MaterialGPU>& materialData
    );

    /// Get the TLAS handle for descriptor binding.
    vk::AccelerationStructureKHR getTLAS() const { return *m_tlas; }

    /// Device address of the TLAS (for shader binding).
    vk::DeviceAddress getTLASAddress() const { return m_tlasAddress; }

    /// Get the material uniform buffer for descriptor binding.
    const GPUBuffer& getMaterialBuffer() const { return m_materialBuffer; }

    /// Number of materials stored in the UBO.
    uint32_t getMaterialCount() const { return m_materialCount; }

private:
    void createCommandPool(uint32_t computeQueueFamily);
    vk::raii::CommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(vk::raii::CommandBuffer& cmdBuf);

    /// Build a single BLAS from mesh data. Returns {blas handle, address, buffer}.
    struct BlasResult {
        vk::raii::AccelerationStructureKHR blas      = nullptr;
        vk::DeviceAddress                  address   = 0;
        GPUBuffer                          buffer;
    };
    BlasResult buildSingleBLAS(const MeshData& mesh);

    void buildTLAS(uint32_t instanceCount);
    void uploadMaterialBuffer(const std::vector<MaterialGPU>& data);

    const vk::raii::Device&          m_device;
    const vk::raii::PhysicalDevice&  m_physDevice;
    uint32_t                         m_queueFamily = 0;

    vk::raii::CommandPool            m_commandPool        = nullptr;

    // BLAS (one or more — Phase 1 keeps one, Phase 2 keeps two)
    std::vector<BlasResult>          m_blasList;

    // TLAS
    vk::raii::AccelerationStructureKHR m_tlas             = nullptr;
    vk::DeviceAddress                   m_tlasAddress     = 0;

    // Temporary build buffers
    GPUBuffer m_scratchBuffer;
    GPUBuffer m_tlasBuffer;
    GPUBuffer m_instanceBuffer;

    // Material uniform buffer
    GPUBuffer m_materialBuffer;
    uint32_t  m_materialCount = 0;
};
