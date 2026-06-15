// All descriptor bindings (set=0) shared across shader modules
import common.types;

public [[vk::binding(0, 0)]]  RaytracingAccelerationStructure tlas;
public [[vk::binding(1, 0)]]  RWTexture2D<float4>             outputImage;

public struct MaterialBlock { public GpuMaterial items[16]; };
public [[vk::binding(2, 0)]]  ConstantBuffer<MaterialBlock>    materials;
public struct LightBlock { public GpuLight items[8]; };
public [[vk::binding(3, 0)]]  ConstantBuffer<LightBlock>       lights;

public [[vk::binding(4, 0)]]  StructuredBuffer<float>          vertexData;
public [[vk::binding(5, 0)]]  StructuredBuffer<uint>           indexData;
public [[vk::binding(12, 0)]] StructuredBuffer<float>          normalData;
public [[vk::binding(13, 0)]] RWStructuredBuffer<float>        accumData;

public [[vk::binding(14, 0)]] RWTexture2D<float4>              normalImage;
public [[vk::binding(15, 0)]] RWTexture2D<float>               depthImage;

static const int MAX_INSTANCES = 16;
public struct InstanceRangeBlock {
    public uint vtxOffset[MAX_INSTANCES];
    public uint vtxCount[MAX_INSTANCES];
    public uint idxOffset[MAX_INSTANCES];
    public uint idxCount[MAX_INSTANCES];
    public uint materialID[MAX_INSTANCES];
    public uint useSmoothNormals[MAX_INSTANCES];
};
public [[vk::binding(6, 0)]]  StructuredBuffer<InstanceRangeBlock>  instanceRanges;

public struct InstanceNormalBlock {
    public float4 normalCol0[MAX_INSTANCES];
    public float4 normalCol1[MAX_INSTANCES];
    public float4 normalCol2[MAX_INSTANCES];
};
public [[vk::binding(16, 0)]] StructuredBuffer<InstanceNormalBlock> instanceNormals;

public [[vk::binding(7, 0)]]  RWStructuredBuffer<PhotonRecord>  photonBuffer;
public [[vk::binding(8, 0)]]  RWStructuredBuffer<uint>         photonCounter;

// Spatial hash map for accelerated photon gather
// Interleaved: [h*2]=count, [h*2+1]=offset (both uint)
public [[vk::binding(9, 0)]]  RWStructuredBuffer<uint>         hashCellData;
public [[vk::binding(10, 0)]] RWStructuredBuffer<uint>         sortedPhotonIndices;

// Per-cell aggregated photon data (raw float array: 23 floats/cell)
// Layout: pos(4) + nrm(4) + dir(4) + spec(10) + photonCount(1 at base+22)
public [[vk::binding(18, 0)]] RWStructuredBuffer<float> cellPhotonData;

// Per-cell pre-gathered output (hash_gather compute → camera RT)
// Layout: 13 uints/cell = 10 spectral + 3 position (asfloat) for hash-collision guard
public [[vk::binding(19, 0)]] RWStructuredBuffer<uint>  gatheredCellData;

// Ray tracing statistics (atomic counters)
public [[vk::binding(17, 0)]] RWStructuredBuffer<uint>         rayStats;

public [[vk::binding(11, 0)]] Texture2D<float4> envMap;
public [[vk::binding(11, 0)]] SamplerState      envSampler;
