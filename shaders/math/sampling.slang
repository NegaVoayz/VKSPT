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
