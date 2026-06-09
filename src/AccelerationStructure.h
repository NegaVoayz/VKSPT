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
    static constexpr uint32_t MAX_MATERIALS = 16;

    /// Input vertex data (positions and normals, interleaved as xyz floats).
    struct MeshData {
        std::vector<float>    vertices;   // interleaved xyz positions
        std::vector<uint32_t> indices;    // triangle indices (3 per triangle)
        std::vector<float>    normals;    // interleaved xyz normals (same indexing as vertices; empty = use geometric)
    };

    /// Descriptor for a single TLAS instance.
    struct InstanceInfo {
        MeshData mesh;
        uint32_t customIndex  = 0;   // rayQueryGetIntersectionInstanceCustomIndexEXT
        uint32_t materialID   = 0;   // index into material UBO
        bool     hasNormals   = false; // per-vertex normals available for smooth interpolation
        float    transform[3][4] = {  // row-major 3×4 affine transform
            {1,0,0,0},
            {0,1,0,0},
            {0,0,1,0}
        };
    };

    /// CPU-side material data uploaded to the GPU UBO.
    /// Must match GpuMaterial in raytrace.comp (std140).
    struct alignas(16) MaterialGPU {
        float cauchyA[4] = {};    // (R, G, B, _) — IOR Cauchy A
        float cauchyB[4] = {};    // (R, G, B, _) — IOR Cauchy B (λ in μm)
        float absorpA[4] = {};    // (R, G, B, _) — absorption Cauchy A: α(λ)=A+B/λ² [m⁻¹]
        float absorpB[4] = {};    // (R, G, B, _) — absorption Cauchy B
        float albedo[4]   = {};   // (R, G, B, _)
        float params[4]   = {};   // (ior, roughness, type, _)
    };

    /// Maximum number of lights in the light UBO.
    static constexpr uint32_t MAX_LIGHTS = 4;

    /// Maximum number of TLAS instances (must match shader).
    static constexpr uint32_t MAX_INSTANCES = 16;

    /// CPU-side light data. Must match GpuLight in raytrace.comp (std140).
    /// Each light holds 16 SPD samples packed as 4 vec4s (64 bytes).
    /// The shader declares `GpuLight items[MAX_LIGHTS]`; we upload exactly
    /// MAX_LIGHTS elements so buffer size matches shader expectation.
    struct alignas(16) GpuLight {
        float spd[16] = {};       // SPD sampled at the 16 reference wavelengths (380–780 nm)
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
        const std::vector<MaterialGPU>& materialData,
        const GpuLight& skyLight
    );

    /// Build N instances with their BLAS + TLAS + materials + lights.
    void buildScene(
        const std::vector<InstanceInfo>& instances,
        const std::vector<MaterialGPU>& materialData,
        const GpuLight& skyLight
    );

    /// Get the TLAS handle for descriptor binding.
    vk::AccelerationStructureKHR getTLAS() const { return *m_tlas; }

    /// Device address of the TLAS (for shader binding).
    vk::DeviceAddress getTLASAddress() const { return m_tlasAddress; }

    /// Get the material uniform buffer for descriptor binding.
    const GPUBuffer& getMaterialBuffer() const { return m_materialBuffer; }

    /// Get the light uniform buffer for descriptor binding.
    const GPUBuffer& getLightBuffer() const { return m_lightBuffer; }

    /// Get the persistent vertex data SSBO for shader normal computation.
    const GPUBuffer& getVertexDataBuffer() const { return m_vertexDataBuffer; }

    /// Get the persistent index data SSBO for shader normal computation.
    const GPUBuffer& getIndexDataBuffer() const { return m_indexDataBuffer; }

    /// Get the persistent normal data SSBO for smooth normal interpolation.
    const GPUBuffer& getNormalDataBuffer() const { return m_normalDataBuffer; }

    /// Get the instance range SSBO for shader vertex lookup.
    const GPUBuffer& getRangeBuffer() const { return m_rangeBuffer; }

    /// Number of TLAS instances.
    uint32_t getInstanceCount() const { return m_instanceCount; }

    /// Number of materials stored in the UBO.
    uint32_t getMaterialCount() const { return m_materialCount; }

    /// Load environment map from a JPEG/PNG file, upload to GPU.
    /// Creates VkImage + VkImageView + VkSampler used as binding 11.
    void loadEnvMap(const std::string& path);

    /// Get the environment map image view.
    vk::ImageView getEnvMapView() const { return *m_envMapView; }

    /// Get the environment map sampler.
    vk::Sampler getEnvMapSampler() const { return *m_envMapSampler; }

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
    void uploadLightBuffer(const GpuLight& skyLight);
    void uploadVertexSSBO();

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

    // Light uniform buffer (sky/environment SPD)
    GPUBuffer m_lightBuffer;

    // Persistent geometry SSBOs (kept alive for shader normal computation)
    GPUBuffer                m_vertexDataBuffer;   // concatenated vertex positions (float3)
    GPUBuffer                m_indexDataBuffer;    // concatenated triangle indices (uint)
    GPUBuffer                m_normalDataBuffer;   // concatenated vertex normals (float3)
    GPUBuffer                m_rangeBuffer;        // per-instance vertex/index ranges
    uint32_t                 m_instanceCount = 0;
    std::vector<std::vector<float>>    m_stagedVertices;   // keep a copy for SSBO
    std::vector<std::vector<uint32_t>> m_stagedIndices;    // keep a copy for SSBO
    std::vector<std::vector<float>>    m_stagedNormals;    // per-instance vertex normal data
    std::vector<std::array<std::array<float,4>,3>> m_instanceTransforms;  // per-instance 3×4

    // Environment map texture
    vk::raii::Image        m_envMapImage    = nullptr;
    vk::raii::DeviceMemory m_envMapMemory   = nullptr;
    vk::raii::ImageView    m_envMapView     = nullptr;
    vk::raii::Sampler      m_envMapSampler  = nullptr;
};
