#pragma once

#include "core/GPUBuffer.h"
#include "ray/AccelBuilder.h"
#include "ray/EnvMap.h"
#include "ray/GeometryBuffer.h"
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <vector>

class AccelerationStructure {
public:
    static constexpr uint32_t MAX_MATERIALS = 16;
    static constexpr uint32_t MAX_LIGHTS    = 4;
    static constexpr uint32_t MAX_INSTANCES = 16;

    struct MeshData {
        std::vector<float>    vertices, normals;
        std::vector<uint32_t> indices;
    };
    struct InstanceInfo {
        MeshData mesh;
        uint32_t customIndex = 0, materialID = 0;
        bool     hasNormals = false;
        float    transform[3][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0}};
    };
    struct alignas(16) MaterialGPU {
        float cauchyA[4]{}, cauchyB[4]{}, absorpA[4]{}, absorpB[4]{};
        float albedo[4]{}, params[4]{};
    };
    struct alignas(16) GpuLight {
        float pos_type[4]{}, color_intensity[4]{};
        float dir_inner[4]{}, outer_range[4]{};
    };

    AccelerationStructure(const vk::raii::Device& dev,
        const vk::raii::PhysicalDevice& physDev, uint32_t qf);
    ~AccelerationStructure();
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure(AccelerationStructure&&) = delete;
    AccelerationStructure& operator=(AccelerationStructure&&) = delete;

    void build(const MeshData& mesh);
    void buildTwoInstance(const InstanceInfo& i0, const InstanceInfo& i1,
        const std::vector<MaterialGPU>& mats, const GpuLight& sky);
    void buildScene(const std::vector<InstanceInfo>& instances,
        const std::vector<MaterialGPU>& mats,
        const std::vector<GpuLight>&    lights);

    vk::AccelerationStructureKHR getTLAS() const { return *m_tlas; }
    vk::DeviceAddress getTLASAddress() const { return m_tlasAddr; }
    const GPUBuffer& getMaterialBuffer() const { return m_matBuf; }
    const GPUBuffer& getLightBuffer()    const { return m_lightBuf; }
    const GeometryBuffer& getGeometry()  const { return m_geom; }
    const GPUBuffer& getInstanceNormalBuffer() const { return m_normBuf; }
    uint32_t getInstanceCount()  const { return m_instCount; }
    uint32_t getMaterialCount()  const { return m_matCount; }

    void loadEnvMap(const std::string& path)
        { m_envMap.load(m_device, m_physDevice, m_qf, path); }
    EnvMap& getEnvMap() { return m_envMap; }
    const EnvMap& getEnvMap() const { return m_envMap; }

private:
    void uploadMaterialBuffer(const std::vector<MaterialGPU>& d);
    void uploadLightBuffer(const std::vector<GpuLight>& l);

    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    uint32_t m_qf = 0;

    AccelBuilder m_builder;
    std::vector<AccelBuilder::BlasResult> m_blasList;
    vk::raii::AccelerationStructureKHR m_tlas = nullptr;
    vk::DeviceAddress m_tlasAddr = 0;
    GPUBuffer m_scratch, m_tlasBuf, m_instBuf;
    GPUBuffer m_matBuf, m_lightBuf;
    uint32_t m_matCount = 0, m_instCount = 0;

    GeometryBuffer m_geom;
    GPUBuffer      m_normBuf;
    std::vector<std::vector<float>>    m_stagedV;
    std::vector<std::vector<uint32_t>> m_stagedI;
    std::vector<std::vector<float>>    m_stagedN;
    std::vector<std::array<std::array<float,4>,3>> m_xfs;
    EnvMap m_envMap;
};
