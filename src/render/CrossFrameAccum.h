#pragma once

#include "core/GPUBuffer.h"
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <cstring>

/// Manages per-pixel RGB+count accumulation across frames for progressive
/// rendering. Detects camera movement to trigger a reset.
class CrossFrameAccum {
public:
    CrossFrameAccum(const vk::raii::Device&         device,
                     const vk::raii::PhysicalDevice& physDevice,
                     uint32_t queueFamily,
                     uint32_t width, uint32_t height);

    /// Zero-fill the accumulation buffer (creates transient cmd buf).
    void reset();

    /// Returns true if the camera moved enough to warrant an accum reset.
    bool detectChange(const float origin[3], const float camU[3],
                      const float camV[3], const float camW[3]);

    const GPUBuffer& Buffer() const { return m_accumBuffer; }
    vk::DeviceSize   BufSize() const { return m_accumBuffer.Size; }
    int              FrameCount() const { return m_accumFrameCount; }
    void             IncFrameCount() { ++m_accumFrameCount; }

private:
    const vk::raii::Device& m_device;
    uint32_t                 m_queueFamily = 0;
    GPUBuffer                m_accumBuffer;
    GPUBuffer                m_accumStaging;
    float                    m_lastOrigin[3] = {};
    float                    m_lastCamU[3]   = {};
    float                    m_lastCamV[3]   = {};
    float                    m_lastCamW[3]   = {};
    int                      m_accumFrameCount = 0;
};
