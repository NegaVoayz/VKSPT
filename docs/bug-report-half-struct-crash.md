# Bug: RWStructuredBuffer<struct-with-half> in compute shader causes driver crash

## Symptom

Process exits with `-1073740791` (`0xC0000409` = `STATUS_STACK_BUFFER_OVERRUN`) during render loop.

## Reproduction

1. Declare a new `RWStructuredBuffer<PhotonRecord>` at a new binding slot (e.g. 18) in `common.bindings.slang`:

```hlsl
// PhotonRecord contains half4 / half2 fields
public [[vk::binding(18, 0)]] RWStructuredBuffer<PhotonRecord> cellPhotonBuffer;
```

2. Create a compute shader that writes to this buffer:

```hlsl
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    PhotonRecord zero;
    zero.phPos = float4(0,0,0,0);
    // ... (set all fields to zero)
    cellPhotonBuffer[tid.x] = zero;
}
```

3. Dispatch the compute pipeline → crash during command buffer execution.

## Isolation

| Test | Result |
|------|--------|
| Compute writes `RWStructuredBuffer<PhotonRecord>` at new binding 18 | **Crash** |
| Compute writes `RWStructuredBuffer<PhotonRecord>` at new binding 18 (no-op: just zero fill) | **Crash** |
| Compute writes `uint` to `RWStructuredBuffer<uint>` at binding 17 (`rayStats`) | OK |
| Compute writes `float4` to `cellPhotonBuffer[hash].phPos` (partial PhotonRecord write) | **Crash** |
| Compute writes raw `RWStructuredBuffer<float>` at binding 18 (19 floats/cell) | OK |
| RT pipeline reads `RWStructuredBuffer<PhotonRecord>` at binding 7 (`photonBuffer`) | OK (pre-existing) |
| Compute reads `RWStructuredBuffer<PhotonRecord>` at binding 7 (scatter/count shaders) | OK (pre-existing) |
| Same pipeline dispatch but with known-good SPV (scatter) | OK |

## Root Cause Hypothesis

The crash happens only when a **new** compute pipeline accesses a **new** binding slot with `RWStructuredBuffer<T>` where `T` contains `half` types (`half4`/`half2`). Pre-existing bindings (slot 7, `photonBuffer`) with the same type work fine in both RT and compute pipelines.

Possible driver-level issues:
1. The SPIR-V module only references one StorageBuffer variable (binding 18) containing half types — the driver may mishandle the sparse binding layout.
2. Vulkan descriptor set compatibility: the pipeline layout's descriptor set layout includes ALL bindings (0–18), but the compute shader's SPIR-V only declares one of them. The driver may fail to match the layout correctly when the sole binding uses 16-bit types.
3. AMD/NVIDIA driver bug triggered by a compute pipeline whose only storage buffer binding contains half types at a non-sequential slot number.

## Workaround

Use `RWStructuredBuffer<float>` and manually pack/unpack the struct fields:

```hlsl
// In bindings — raw float array
public [[vk::binding(18, 0)]] RWStructuredBuffer<float> cellPhotonData;

// Layout per cell (19 floats):
//   [0..3]   position (xyz + pad)
//   [4..7]   normal (xyz + pad)
//   [8..11]  direction (xyz + pad)
//   [12..21] spectral buckets 0..9 (already summed, float precision)
//   [22]     photon count (as float, cast to uint when reading)

// Write (aggregate):
uint base = hash * 19;
cellPhotonData[base + 0] = posX;
// ...

// Read (gather):
float3 phPos = float3(cellPhotonData[base+0], cellPhotonData[base+1], cellPhotonData[base+2]);
float sb0 = cellPhotonData[base + 12 + 0];
uint cellCount = uint(cellPhotonData[base + 18]);
```

## Environment

- Vulkan SDK 1.4.341.1
- Slang compiler (spirv target)
- GPU: (user's GPU)
- Enabled features: `storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess`, `shaderFloat16`, `bufferDeviceAddress`
