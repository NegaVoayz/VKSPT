#include "RaySorter.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

RaySorter::RaySorter(const vk::raii::Device&         device,
                     const vk::raii::PhysicalDevice& physDevice,
                     uint32_t width, uint32_t height, uint32_t spp)
    : m_device(device)
    , m_physDevice(physDevice)
    , m_width(width)
    , m_height(height)
    , m_spp(spp)
{
    // --- Ray buffer (global, device-local) ---
    vk::DeviceSize rayBufSize = MAX_RAYS * sizeof(PackedRay);
    m_rayBuffer = GPUBuffer::create(
        m_device, rayBufSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );

    // --- Counter buffer (head/tail, 16 bytes) ---
    vk::DeviceSize counterSize = sizeof(CounterData);
    m_counterBuffer = GPUBuffer::create(
        m_device, counterSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );
    m_counterStaging = GPUBuffer::create(
        m_device, counterSize,
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );

    // --- Pixel accumulator (per-pixel PixelEntry, 24 bytes per pixel) ---
    uint32_t pixelCount = m_width * m_height;
    vk::DeviceSize accumSize = pixelCount * sizeof(PixelEntry);
    m_accumBuffer = GPUBuffer::create(
        m_device, accumSize,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );
    m_accumStaging = GPUBuffer::create(
        m_device, accumSize,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );
}

RaySorter::~RaySorter() {
    // vk::raii / GPUBuffer handles cleanup
}

void RaySorter::initRays(const float* camOrigin, const float* camU,
                          const float* camV, const float* camW, float fovTan) {
    uint32_t pixelCount = m_width * m_height;
    uint32_t totalRays = pixelCount * m_spp;
    m_activeRayCount = std::min(totalRays, MAX_RAYS);

    std::vector<PackedRay> rays(m_activeRayCount);

    for (uint32_t py = 0; py < m_height; ++py) {
        for (uint32_t px = 0; px < m_width; ++px) {
            uint32_t base = (py * m_width + px) * m_spp;
            for (uint32_t s = 0; s < m_spp && base + s < m_activeRayCount; ++s) {
                PackedRay& r = rays[base + s];

                uint32_t h = px * 1973u + py * 9277u + s * 2654435761u;
                float jx = float(h & 0xFFFFu) / 65536.0f - 0.5f;
                float jy = float((h >> 16u) & 0xFFFFu) / 65536.0f - 0.5f;
                float u = (float(px) + 0.5f + jx) / float(m_width) * 2.0f - 1.0f;
                float v = (float(py) + 0.5f + jy) / float(m_height) * 2.0f - 1.0f;
                v = -v;

                float dx = camW[0] + u * camU[0] + v * camV[0];
                float dy = camW[1] + u * camU[1] + v * camV[1];
                float dz = camW[2] + u * camU[2] + v * camV[2];
                float invLen = 1.0f / sqrtf(dx*dx + dy*dy + dz*dz);

                r.origin[0] = camOrigin[0];
                r.origin[1] = camOrigin[1];
                r.origin[2] = camOrigin[2];
                r.lamStart   = 380.0f;
                r.direction[0] = dx * invLen;
                r.direction[1] = dy * invLen;
                r.direction[2] = dz * invLen;
                r.lamEnd     = 780.0f;
                r.lastSplit[0] = 0.0f;
                r.lastSplit[1] = 0.0f;
                r.lastSplit[2] = 0.0f;
                r.energy     = 1.0f / float(m_spp);  // total per-pixel = 1.0 across all SPP samples
                r.dispersion[0] = 0.0f;
                r.dispersion[1] = 0.0f;
                r.dispersion[2] = 0.0f;
                r.bounce     = 0.0f;
                r.generation = 0;
                r.pixelIndex = int(py * m_width + px);
                r.rayAction  = 0;
                r.insideGlass    = 0;
                r.fromReflection = 0;
                r.hadTIR         = 0;
                r.hadFresnelRefl = 0;
                r._pad = 0;
            }
        }
    }

    // Upload via staging
    vk::DeviceSize uploadSize = m_activeRayCount * sizeof(PackedRay);
    auto staging = GPUBuffer::createStaging(
        m_device, rays.data(), uploadSize,
        vk::BufferUsageFlagBits::eTransferSrc, m_physDevice
    );

    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cmdBuf = cmdBufs[0];

    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, 0, uploadSize);
    cmdBuf.copyBuffer(*staging.buffer, *m_rayBuffer.buffer, region);
    cmdBuf.end();

    auto queue = m_device.getQueue(0, 0);
    vk::SubmitInfo submit({}, {}, *cmdBuf);
    queue.submit(submit, nullptr);
    queue.waitIdle();

    // Set initial counters: head=0, tail=activeRayCount
    resetCounters();
}

void RaySorter::resetCounters() {
    CounterData cd{};
    cd.head = 0;
    cd.tail = m_activeRayCount;

    void* mapped = m_counterStaging.memory.mapMemory(0, sizeof(CounterData));
    std::memcpy(mapped, &cd, sizeof(CounterData));
    m_counterStaging.memory.unmapMemory();

    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cmdBuf = cmdBufs[0];

    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, 0, sizeof(CounterData));
    cmdBuf.copyBuffer(*m_counterStaging.buffer, *m_counterBuffer.buffer, region);
    cmdBuf.end();

    auto queue = m_device.getQueue(0, 0);
    vk::SubmitInfo submit({}, {}, *cmdBuf);
    queue.submit(submit, nullptr);
    queue.waitIdle();
}

void RaySorter::advanceHead(uint32_t newHead) {
    // Upload new head value, keep tail as-is
    void* mapped = m_counterStaging.memory.mapMemory(0, sizeof(CounterData));
    CounterData cd;
    std::memcpy(&cd, mapped, sizeof(CounterData));
    cd.head = newHead;
    std::memcpy(mapped, &cd, sizeof(CounterData));
    m_counterStaging.memory.unmapMemory();

    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cmdBuf = cmdBufs[0];

    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, 0, sizeof(uint32_t));  // only copy head
    cmdBuf.copyBuffer(*m_counterStaging.buffer, *m_counterBuffer.buffer, region);
    cmdBuf.end();

    auto queue = m_device.getQueue(0, 0);
    vk::SubmitInfo submit({}, {}, *cmdBuf);
    queue.submit(submit, nullptr);
    queue.waitIdle();
}

uint32_t RaySorter::getTailCount() {
    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cmdBuf = cmdBufs[0];

    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, 0, sizeof(CounterData));
    cmdBuf.copyBuffer(*m_counterBuffer.buffer, *m_counterStaging.buffer, region);
    cmdBuf.end();

    auto queue = m_device.getQueue(0, 0);
    vk::SubmitInfo submit({}, {}, *cmdBuf);
    queue.submit(submit, nullptr);
    queue.waitIdle();

    void* mapped = m_counterStaging.memory.mapMemory(0, sizeof(CounterData));
    CounterData cd;
    std::memcpy(&cd, mapped, sizeof(CounterData));
    m_counterStaging.memory.unmapMemory();

    return cd.tail;
}

void RaySorter::readbackAccumulator(void* output, uint32_t pixelCount) {
    vk::DeviceSize accumSize = pixelCount * sizeof(PixelEntry);

    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cmdBuf = cmdBufs[0];

    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, 0, accumSize);
    cmdBuf.copyBuffer(*m_accumBuffer.buffer, *m_accumStaging.buffer, region);
    cmdBuf.end();

    auto queue = m_device.getQueue(0, 0);
    vk::SubmitInfo submit({}, {}, *cmdBuf);
    queue.submit(submit, nullptr);
    queue.waitIdle();

    void* mapped = m_accumStaging.memory.mapMemory(0, accumSize);
    std::memcpy(output, mapped, static_cast<size_t>(accumSize));
    m_accumStaging.memory.unmapMemory();
}
