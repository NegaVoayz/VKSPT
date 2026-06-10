// Geometric normal computation from vertex SSBO data

vec3 computeGeometricNormal(int instanceID, int primitiveID) {
    uint vOff = instanceRanges.vtxOffset[instanceID];
    uint iOff = instanceRanges.idxOffset[instanceID];
    uint i0 = indexData.i[iOff + uint(primitiveID) * 3u + 0u];
    uint i1 = indexData.i[iOff + uint(primitiveID) * 3u + 1u];
    uint i2 = indexData.i[iOff + uint(primitiveID) * 3u + 2u];
    vec3 v0 = vec3(vertexData.v[vOff + i0 * 3u + 0u],
                   vertexData.v[vOff + i0 * 3u + 1u],
                   vertexData.v[vOff + i0 * 3u + 2u]);
    vec3 v1 = vec3(vertexData.v[vOff + i1 * 3u + 0u],
                   vertexData.v[vOff + i1 * 3u + 1u],
                   vertexData.v[vOff + i1 * 3u + 2u]);
    vec3 v2 = vec3(vertexData.v[vOff + i2 * 3u + 0u],
                   vertexData.v[vOff + i2 * 3u + 1u],
                   vertexData.v[vOff + i2 * 3u + 2u]);
    vec3 e0 = v1 - v0;
    vec3 e1 = v2 - v0;
    return normalize(cross(e0, e1));
}

vec3 computeSmoothNormal(int instanceID, int primitiveID, vec2 bary) {
    uint vOff = instanceRanges.vtxOffset[instanceID];
    uint iOff = instanceRanges.idxOffset[instanceID];
    uint i0 = indexData.i[iOff + uint(primitiveID) * 3u + 0u];
    uint i1 = indexData.i[iOff + uint(primitiveID) * 3u + 1u];
    uint i2 = indexData.i[iOff + uint(primitiveID) * 3u + 2u];
    vec3 n0 = vec3(normalData.n[vOff + i0 * 3u + 0u],
                   normalData.n[vOff + i0 * 3u + 1u],
                   normalData.n[vOff + i0 * 3u + 2u]);
    vec3 n1 = vec3(normalData.n[vOff + i1 * 3u + 0u],
                   normalData.n[vOff + i1 * 3u + 1u],
                   normalData.n[vOff + i1 * 3u + 2u]);
    vec3 n2 = vec3(normalData.n[vOff + i2 * 3u + 0u],
                   normalData.n[vOff + i2 * 3u + 1u],
                   normalData.n[vOff + i2 * 3u + 2u]);
    float u = bary.x, v = bary.y;
    float w = 1.0 - u - v;
    return normalize(n0 * w + n1 * u + n2 * v);
}
