#pragma once

#include "GPUBuffer.h"
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <vector>

/// Manages the global ray buffer and sorting infrastructure for the Phase 4
/// multi-pass ray tracing pipeline. Replaces the per-pixel stack with a flat
/// array of rays processed sequentially across dispatches.
///
/// Pipeline:
///   1. Init ray buffer (CPU fills with one ray per pixel × SPP)
///   2. Loop until active ray count is 0:
///      a. classify.comp — ray query + action classification
///      b. process.comp — per-action processing (one indirect dispatch per action)
class RaySorter {
public:
    /// Maximum number of rays in the global buffer.
    /// For 800×600×4 SPP = 1.92M initial rays, with splits up to ~4×:
    /// 8M rays × 80 bytes = 640 MB. We use 4M (320 MB) and drop overflow.
    static constexpr uint32_t MAX_RAYS = 16 * 1024 * 1024; // 16M rays (1.5GB)
    static constexpr uint32_t OVERFLOW_SIZE = 2 * 1024 * 1024; // 2M ray overflow (192MB, host-visible)

    /// Number of action buckets.
    static constexpr uint32_t ACTION_COUNT = 6;

    /// RayAction enum values (match GLSL classify shader).
    enum RayAction : int {
        ACTION_UNPROCESSED  = 0,
        ACTION_MISS         = 1,
        ACTION_DIELECTRIC   = 2,
        ACTION_LAMBERTIAN   = 3,
        ACTION_METAL        = 4,
        ACTION_DEAD         = 5,
        ACTION_TERMINATED   = 6,
    };

    /// Packed ray structure matching the GLSL std430 layout.
    /// Total: 80 bytes
    struct alignas(16) PackedRay {
        float origin[3];
        float lamStart;           // stored as float, cast to int in shader
        float direction[3];
        float lamEnd;
        float lastSplit[3];
        float energy;
        float dispersion[3];
        float bounce;             // stored as float
        int   generation;
        int   pixelIndex;         // which output pixel this ray contributes to
        int   rayAction;          // 0=UNPROCESSED, 1=MISS, 2=DIELECTRIC, 3=LAMBERTIAN, 4=METAL, 5=DEAD
        int   insideGlass;        // bool as int
        int   fromReflection;
        int   hadTIR;
        int   hadFresnelRefl;
        int   _pad;               // align to 96 bytes (6 × 16)
    };
    static_assert(sizeof(PackedRay) == 96, "PackedRay must match shader layout");

    /// GPU-side counter buffer layout (matches shader CounterBlock).
    /// head: start of active ray range (set by CPU between dispatches)
    /// tail: next write position (atomic counter for new ray appends)
    struct alignas(16) CounterData {
        uint32_t head;
        uint32_t tail;
        float    _pad[2];   // pad to 16 bytes
    };
    static_assert(sizeof(CounterData) == 16, "CounterData size mismatch");

    /// Per-pixel accumulator entry (matches shader PixelEntry, 6 uints).
    struct alignas(4) PixelEntry {
        uint32_t colorR, colorG, colorB;
        uint32_t wlR, wlG, wlB;
    };
    static_assert(sizeof(PixelEntry) == 24, "PixelEntry size mismatch");

    RaySorter(const vk::raii::Device& device,
              const vk::raii::PhysicalDevice& physDevice,
              uint32_t width, uint32_t height, uint32_t spp);
    ~RaySorter();

    RaySorter(const RaySorter&)            = delete;
    RaySorter& operator=(const RaySorter&) = delete;
    RaySorter(RaySorter&&)                 = delete;
    RaySorter& operator=(RaySorter&&)      = delete;

    /// Fill the ray buffer with initial camera rays (called once per frame).
    void initRays(const float* camOrigin, const float* camU, const float* camV,
                  const float* camW, float fovTan);

    /// Get the global ray buffer for descriptor binding.
    const GPUBuffer& getRayBuffer() const { return m_rayBuffer; }

    /// Get the action counter buffer for descriptor binding.
    const GPUBuffer& getCounterBuffer() const { return m_counterBuffer; }

    /// Get the pixel accumulator buffer for descriptor binding.
    const GPUBuffer& getAccumBuffer() const { return m_accumBuffer; }

    /// Upload reset counters to GPU (head=0, tail=activeRayCount).
    void resetCounters();

    /// Advance the head pointer (processed up to this index).
    void advanceHead(uint32_t newHead);

    /// Read back tail (next write position) from GPU.
    uint32_t getTailCount();

    /// Read back pixel accumulator for final output.
    void readbackAccumulator(void* output, uint32_t pixelCount);

    /// Sort a batch of rays [head, head+count) by rayAction on CPU.
    /// Readback → std::sort → upload. Batch must be ≤ BATCH_CAP.
    void sortBatchByAction(uint32_t head, uint32_t count);

    /// Reset the GPU tail counter. Call after drainOverflow to clamp tail
    /// past MAX_RAYS (overflow spawns still increment the atomic counter).
    void clampTail();

    /// Drain overflow buffer to host stash. Call after each dispatch.
    /// Returns number of rays stashed this cycle.
    uint32_t drainOverflow();

    /// Inject stashed rays back into the GPU ray buffer (up to maxCount).
    /// Returns number of rays injected.
    uint32_t injectStashed(uint32_t maxCount);

    /// Number of rays currently stashed on host.
    uint32_t stashedCount() const { return uint32_t(m_hostStash.size()); }

    /// Get overflow buffer for descriptor binding.
    const GPUBuffer& getOverflowBuffer() const { return m_overflowBuf; }

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    uint32_t getSPP() const { return m_spp; }

private:
    const vk::raii::Device&         m_device;
    const vk::raii::PhysicalDevice& m_physDevice;
    uint32_t m_width, m_height, m_spp;

    // Staging buffer for counter reset + readback
    GPUBuffer m_counterStaging;
    GPUBuffer m_counterBuffer;

    // Ray buffer (output of init + per-bounce processing)
    GPUBuffer m_rayBuffer;
    uint32_t  m_activeRayCount = 0;

    // Pixel accumulator (framebuffer-sized, read back for final output)
    GPUBuffer m_accumBuffer;
    GPUBuffer m_accumStaging;

    // Batch staging buffer for CPU-side rayAction sorting (BATCH_CAP rays)
    GPUBuffer m_batchStaging;
    GPUBuffer m_batchReadback;

    // Overflow: host-visible buffer for shader spill, drained to CPU after dispatch
    GPUBuffer m_overflowBuf;      // host-visible, [counter+pad][PackedRay array]
    std::vector<PackedRay> m_hostStash; // CPU-side stash of overflow rays
};
