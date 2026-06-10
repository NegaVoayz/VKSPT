#include "render/CrossFrameAccum.h"
#include <algorithm>
#include <cmath>

void CrossFrameAccum::init(
    const vk::raii::Device&         device,
    const vk::raii::PhysicalDevice& physDevice,
    uint32_t                        queueFamily,
    uint32_t                        width, uint32_t height)
{
    m_device      = &device;
    m_queueFamily = queueFamily;
    vk::DeviceSize accumSize = width * height * 4 * sizeof(float);
    m_accumBuffer = GPUBuffer::create(
        device, accumSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, physDevice);
    m_accumStaging = GPUBuffer::create(
        device, accumSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent, physDevice);

    void* mapped = m_accumStaging.memory.mapMemory(0, accumSize);
    std::memset(mapped, 0, static_cast<size_t>(accumSize));
    m_accumStaging.memory.unmapMemory();

    auto cmdPool = vk::raii::CommandPool(*m_device,
        vk::CommandPoolCreateInfo(
            vk::CommandPoolCreateFlagBits::eTransient, m_queueFamily));
    auto cmdBufs = vk::raii::CommandBuffers(*m_device,
        vk::CommandBufferAllocateInfo(
            *cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    cmdBufs[0].begin(vk::CommandBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, 0, accumSize);
    cmdBufs[0].copyBuffer(
        *m_accumStaging.buffer, *m_accumBuffer.buffer, region);
    cmdBufs[0].end();
    auto q = m_device->getQueue(m_queueFamily, 0);
    q.submit(vk::SubmitInfo({}, {}, *cmdBufs[0]), nullptr);
    q.waitIdle();
}

void CrossFrameAccum::reset()
{
    vk::DeviceSize sz = m_accumBuffer.size;
    void* mapped = m_accumStaging.memory.mapMemory(0, sz);
    std::memset(mapped, 0, static_cast<size_t>(sz));
    m_accumStaging.memory.unmapMemory();

    auto cmdPool = vk::raii::CommandPool(*m_device,
        vk::CommandPoolCreateInfo(
            vk::CommandPoolCreateFlagBits::eTransient, m_queueFamily));
    auto cmdBufs = vk::raii::CommandBuffers(*m_device,
        vk::CommandBufferAllocateInfo(
            *cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    cmdBufs[0].begin(vk::CommandBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, 0, sz);
    cmdBufs[0].copyBuffer(
        *m_accumStaging.buffer, *m_accumBuffer.buffer, region);
    cmdBufs[0].end();
    auto q = m_device->getQueue(m_queueFamily, 0);
    q.submit(vk::SubmitInfo({}, {}, *cmdBufs[0]), nullptr);
    q.waitIdle();
    m_accumFrameCount = 0;
}

bool CrossFrameAccum::detectChange(
    const float origin[3], const float camU[3],
    const float camV[3], const float camW[3])
{
    const float eps = 0.001f;
    auto moved = [eps](const float* a, const float* b) {
        return std::abs(a[0] - b[0]) > eps ||
               std::abs(a[1] - b[1]) > eps ||
               std::abs(a[2] - b[2]) > eps;
    };
    bool changed = moved(origin, m_lastOrigin) ||
                   moved(camU, m_lastCamU) ||
                   moved(camV, m_lastCamV) ||
                   moved(camW, m_lastCamW);
    std::copy(origin, origin + 3, m_lastOrigin);
    std::copy(camU, camU + 3, m_lastCamU);
    std::copy(camV, camV + 3, m_lastCamV);
    std::copy(camW, camW + 3, m_lastCamW);
    return changed;
}
