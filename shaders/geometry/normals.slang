// Geometric normal computation from vertex SSBO data

static const int MAX_INSTANCES = 16;
struct InstanceRangeBlock {
    uint vtxOffset[MAX_INSTANCES];
    uint vtxCount[MAX_INSTANCES];
    uint idxOffset[MAX_INSTANCES];
    uint idxCount[MAX_INSTANCES];
    uint materialID[MAX_INSTANCES];
    uint useSmoothNormals[MAX_INSTANCES];
};
[[vk::binding(6, 0)]] StructuredBuffer<InstanceRangeBlock> instanceRanges;
[[vk::binding(5, 0)]]  StructuredBuffer<uint>  indexData;
[[vk::binding(4, 0)]]  StructuredBuffer<float> vertexData;
[[vk::binding(12, 0)]] StructuredBuffer<float> normalData;

public float3 computeGeometricNormal(int instanceID, int primitiveID) {
    uint vOff = instanceRanges[0].vtxOffset[instanceID];
    uint iOff = instanceRanges[0].idxOffset[instanceID];
    uint i0 = indexData[iOff + uint(primitiveID) * 3u + 0u];
    uint i1 = indexData[iOff + uint(primitiveID) * 3u + 1u];
    uint i2 = indexData[iOff + uint(primitiveID) * 3u + 2u];
    float3 v0 = float3(vertexData[vOff + i0 * 3u + 0u],
                       vertexData[vOff + i0 * 3u + 1u],
                       vertexData[vOff + i0 * 3u + 2u]);
    float3 v1 = float3(vertexData[vOff + i1 * 3u + 0u],
                       vertexData[vOff + i1 * 3u + 1u],
                       vertexData[vOff + i1 * 3u + 2u]);
    float3 v2 = float3(vertexData[vOff + i2 * 3u + 0u],
                       vertexData[vOff + i2 * 3u + 1u],
                       vertexData[vOff + i2 * 3u + 2u]);
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    return normalize(cross(e0, e1));
}

public float3 computeSmoothNormal(int instanceID, int primitiveID, float2 bary) {
    uint vOff = instanceRanges[0].vtxOffset[instanceID];
    uint iOff = instanceRanges[0].idxOffset[instanceID];
    uint i0 = indexData[iOff + uint(primitiveID) * 3u + 0u];
    uint i1 = indexData[iOff + uint(primitiveID) * 3u + 1u];
    uint i2 = indexData[iOff + uint(primitiveID) * 3u + 2u];
    float3 n0 = float3(normalData[vOff + i0 * 3u + 0u],
                       normalData[vOff + i0 * 3u + 1u],
                       normalData[vOff + i0 * 3u + 2u]);
    float3 n1 = float3(normalData[vOff + i1 * 3u + 0u],
                       normalData[vOff + i1 * 3u + 1u],
                       normalData[vOff + i1 * 3u + 2u]);
    float3 n2 = float3(normalData[vOff + i2 * 3u + 0u],
                       normalData[vOff + i2 * 3u + 1u],
                       normalData[vOff + i2 * 3u + 2u]);
    float u = bary.x, v = bary.y;
    float w = 1.0 - u - v;
    return normalize(n0 * w + n1 * u + n2 * v);
}
