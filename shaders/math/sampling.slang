// RNG (Wang hash) + cosine-weighted hemisphere sampling

public uint wangHash(uint s) {
    s = (s ^ 61u) ^ (s >> 16u);
    s *= 9u;
    s = s ^ (s >> 4u);
    s *= 0x27d4eb2du;
    s = s ^ (s >> 15u);
    return s;
}

public float rand1(inout uint seed) {
    seed = wangHash(seed);
    return float(seed >> 9u) / 8388608.0;
}

public float3 cosineSampleHemisphere(float3 N, inout uint seed) {
    float r1 = rand1(seed);
    float r2 = rand1(seed);
    float r  = sqrt(r1);
    float theta = 6.283185307 * r2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(1.0 - r1, 0.0));
    float3 T = (abs(N.x) > 0.1 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0));
    float3 B = normalize(cross(N, T));
    float3 U = cross(B, N);
    return normalize(U * x + B * y + N * z);
}

// Orthonormal basis from a unit direction
public void orthonormalBasis(float3 n, out float3 u, out float3 v) {
    float3 t = (abs(n.x) > 0.1 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0));
    u = normalize(cross(n, t));
    v = cross(u, n);
}

// Uniform sample on unit sphere
public float3 uniformSampleSphere(inout uint seed) {
    float z = 1.0 - 2.0 * rand1(seed);
    float r = sqrt(max(1.0 - z * z, 0.0));
    float theta = 6.283185307 * rand1(seed);
    return float3(r * cos(theta), r * sin(theta), z);
}

// Uniform sample in a cone of half-angle cos(angle) around dir
public float3 sampleCone(float3 dir, float cosHalfAngle, inout uint seed) {
    float z = cosHalfAngle + (1.0 - cosHalfAngle) * rand1(seed);
    float r = sqrt(max(1.0 - z * z, 0.0));
    float theta = 6.283185307 * rand1(seed);
    float3 localDir = float3(r * cos(theta), r * sin(theta), z);
    float3 u, v;
    orthonormalBasis(dir, u, v);
    return u * localDir.x + v * localDir.y + dir * localDir.z;
}
