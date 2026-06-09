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
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
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

    // --- Batch staging for CPU-side rayAction sorting ---
    constexpr uint32_t BATCH_CAP = 128 * 1024;
    vk::DeviceSize batchSize = BATCH_CAP * sizeof(PackedRay);
    m_batchStaging = GPUBuffer::create(
        m_device, batchSize,
        vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_physDevice
    );
    m_batchReadback = GPUBuffer::create(
        m_device, batchSize,
        vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );

    // --- Overflow buffer: host-visible, shader spills excess children here ---
    // Layout: [uint32_t overflowTail + 12B pad][PackedRay array]
    vk::DeviceSize overflowBytes = 16 + OVERFLOW_SIZE * sizeof(PackedRay);
    m_overflowBuf = GPUBuffer::create(
        m_device, overflowBytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        m_physDevice
    );
    // Zero-initialize the overflow counter
    void* ovMap = m_overflowBuf.memory.mapMemory(0, 16);
    uint32_t zero = 0;
    std::memcpy(ovMap, &zero, 4);
    m_overflowBuf.memory.unmapMemory();

    m_hostStash.reserve(OVERFLOW_SIZE * 4);  // pre-allocate for up to 4× overflow
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
                r.throughput = 1.0f / float(m_spp);  // total per-pixel = 1.0 across all SPP samples
                r.dispersion[0] = 0.0f;
                r.dispersion[1] = 0.0f;
                r.dispersion[2] = 0.0f;
                r.bounce     = 0.0f;
                r.generation = 0;
                r.pixelIndex = int(py * m_width + px);
                r.rayAction  = 0;
                r.insideGlass    = 0;
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
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
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
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
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
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
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
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
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
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
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

void RaySorter::sortBatchByAction(uint32_t head, uint32_t count) {
    if (count == 0) return;
    vk::DeviceSize batchBytes = count * sizeof(PackedRay);

    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cmdBuf = cmdBufs[0];
    auto queue = m_device.getQueue(0, 0);

    // 1. Copy batch [head, head+count) from ray buffer to batch staging
    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy toStaging(head * sizeof(PackedRay), 0, batchBytes);
    cmdBuf.copyBuffer(*m_rayBuffer.buffer, *m_batchStaging.buffer, toStaging);
    cmdBuf.end();
    vk::SubmitInfo submit1({}, {}, *cmdBuf);
    queue.submit(submit1, nullptr);
    queue.waitIdle();

    // 2. Copy staging → readback (host-visible)
    cmdBufs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy toReadback(0, 0, batchBytes);
    cmdBuf.copyBuffer(*m_batchStaging.buffer, *m_batchReadback.buffer, toReadback);
    cmdBuf.end();
    vk::SubmitInfo submit2({}, {}, *cmdBuf);
    queue.submit(submit2, nullptr);
    queue.waitIdle();

    // 3. Sort on CPU by rayAction
    void* mapped = m_batchReadback.memory.mapMemory(0, batchBytes);
    std::vector<PackedRay> sorted(count);
    std::memcpy(sorted.data(), mapped, static_cast<size_t>(batchBytes));
    m_batchReadback.memory.unmapMemory();

    std::sort(sorted.begin(), sorted.end(),
        [](const PackedRay& a, const PackedRay& b) {
            return a.rayAction < b.rayAction;
        });

    // 4. Upload sorted batch: readback → staging
    void* stagingMapped = m_batchReadback.memory.mapMemory(0, batchBytes);
    std::memcpy(stagingMapped, sorted.data(), static_cast<size_t>(batchBytes));
    m_batchReadback.memory.unmapMemory();

    cmdBufs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy toStaging2(0, 0, batchBytes);
    cmdBuf.copyBuffer(*m_batchReadback.buffer, *m_batchStaging.buffer, toStaging2);
    cmdBuf.end();
    vk::SubmitInfo submit3({}, {}, *cmdBuf);
    queue.submit(submit3, nullptr);
    queue.waitIdle();

    // 5. Copy staging back to ray buffer
    cmdBufs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy toRayBuf(0, head * sizeof(PackedRay), batchBytes);
    cmdBuf.copyBuffer(*m_batchStaging.buffer, *m_rayBuffer.buffer, toRayBuf);
    cmdBuf.end();
    vk::SubmitInfo submit4({}, {}, *cmdBuf);
    queue.submit(submit4, nullptr);
    queue.waitIdle();
}

void RaySorter::clampTail() {
    // Read current tail from GPU
    uint32_t gpuTail = getTailCount();
    if (gpuTail <= MAX_RAYS) return;  // no overflow, no need to clamp

    // Write MAX_RAYS as the new tail (overflow children are stashed, not in ray buf)
    void* mapped = m_counterStaging.memory.mapMemory(0, sizeof(CounterData));
    CounterData cd;
    std::memcpy(&cd, mapped, sizeof(CounterData));
    cd.tail = MAX_RAYS;
    std::memcpy(mapped, &cd, sizeof(CounterData));
    m_counterStaging.memory.unmapMemory();

    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));
    auto& cmdBuf = cmdBufs[0];

    // Copy tail field (offset 4, size 4) to counter buffer
    cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy rgn(4, 4, 4);  // src offset 4 (tail), dst offset 4, size 4
    cmdBuf.copyBuffer(*m_counterStaging.buffer, *m_counterBuffer.buffer, rgn);
    cmdBuf.end();

    auto queue = m_device.getQueue(0, 0);
    vk::SubmitInfo subInfo({}, {}, *cmdBuf);
    queue.submit(subInfo, nullptr);
    queue.waitIdle();
}

uint32_t RaySorter::drainOverflow() {
    // The overflow buffer is host-visible+coherent — read counter directly
    void* mapped = m_overflowBuf.memory.mapMemory(0, 16);
    uint32_t count = 0;
    std::memcpy(&count, mapped, 4);
    if (count == 0) { m_overflowBuf.memory.unmapMemory(); return 0; }
    if (count > OVERFLOW_SIZE) count = OVERFLOW_SIZE;

    // Read overflow rays (offset 16 = after counter header)
    void* rayData = static_cast<char*>(mapped) + 16;
    size_t oldSize = m_hostStash.size();
    m_hostStash.resize(oldSize + count);
    std::memcpy(m_hostStash.data() + oldSize, rayData, count * sizeof(PackedRay));

    // Reset overflow counter to 0 (in-place, host-visible)
    uint32_t zero = 0;
    std::memcpy(mapped, &zero, 4);
    m_overflowBuf.memory.unmapMemory();

    return count;
}

uint32_t RaySorter::injectStashed(uint32_t maxCount) {
    if (m_hostStash.empty() || maxCount == 0) return 0;

    uint32_t count = std::min(maxCount, uint32_t(m_hostStash.size()));
    vk::DeviceSize bytes = count * sizeof(PackedRay);

    // Upload via staging
    auto staging = GPUBuffer::createStaging(m_device, m_hostStash.data(), bytes,
        vk::BufferUsageFlagBits::eTransferSrc, m_physDevice);

    auto queue = m_device.getQueue(0, 0);
    auto cmdPool = vk::raii::CommandPool(m_device,
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient |
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer, 0));
    auto cmdBufs = vk::raii::CommandBuffers(m_device,
        vk::CommandBufferAllocateInfo(*cmdPool, vk::CommandBufferLevel::ePrimary, 1));

    // Read current tail, copy stashed rays to tail position
    uint32_t tail = getTailCount();
    if (tail > MAX_RAYS) tail = MAX_RAYS;
    if (tail + count > MAX_RAYS) count = MAX_RAYS - tail;
    if (count == 0) return 0;

    cmdBufs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy region(0, tail * sizeof(PackedRay), count * sizeof(PackedRay));
    cmdBufs[0].copyBuffer(*staging.buffer, *m_rayBuffer.buffer, region);
    cmdBufs[0].end();
    vk::SubmitInfo si({}, {}, *cmdBufs[0]);
    queue.submit(si, nullptr);
    queue.waitIdle();

    // Update tail counter
    CounterData cd{};
    cd.head = 0;  // unused in this upload
    cd.tail = tail + count;
    void* mapped = m_counterStaging.memory.mapMemory(0, sizeof(CounterData));
    std::memcpy(mapped, &cd, sizeof(CounterData));
    m_counterStaging.memory.unmapMemory();
    cmdBufs[0].begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    vk::BufferCopy cr(0, 0, 16);
    cmdBufs[0].copyBuffer(*m_counterStaging.buffer, *m_counterBuffer.buffer, cr);
    cmdBufs[0].end();
    vk::SubmitInfo si2({}, {}, *cmdBufs[0]);
    queue.submit(si2, nullptr);
    queue.waitIdle();

    // Remove from stash
    m_hostStash.erase(m_hostStash.begin(), m_hostStash.begin() + count);
    return count;
}
