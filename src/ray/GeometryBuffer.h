#pragma once

#include "core/GPUBuffer.h"
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <vector>

/// Owns and uploads the concatenated geometry SSBOs (vertex, index,
/// normal, instance range) used by the shader for normal computation.
class GeometryBuffer {
public:
    uint32_t maxInstances = 16;

    void upload(const vk::raii::Device& dev,
                const vk::raii::PhysicalDevice& physDev,
                const std::vector<std::vector<float>>&    stagedVertices,
                const std::vector<std::vector<uint32_t>>& stagedIndices,
                const std::vector<std::vector<float>>&    stagedNormals,
                const std::vector<uint32_t>& matIDs,
                const std::vector<uint32_t>& smoothFlags);

    const GPUBuffer& vertexBuf() const { return m_vertex; }
    const GPUBuffer& indexBuf()  const { return m_index; }
    const GPUBuffer& normalBuf() const { return m_normal; }
    const GPUBuffer& rangeBuf()  const { return m_range; }

private:
    GPUBuffer m_vertex, m_index, m_normal, m_range;
};
