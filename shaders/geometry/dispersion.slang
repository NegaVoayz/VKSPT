// Dispersion & split utilities
import math.spectral_math;

public float3 computeDispersion(float3 I, float3 N, float eta_mid, float deta) {
    float3 T0, T1;
    float h = deta * 1e-3;
    if (abs(h) < 1e-9) return float3(0);
    if (!refractDir(I, N, eta_mid, T0)) return float3(0);
    if (!refractDir(I, N, eta_mid + h, T1)) return float3(0);
    float3 dT_deta = (T1 - T0) / h;
    float3 spread = dT_deta * deta;
    return spread - dot(spread, T0) * T0;
}

public float3 rotateToNewFrame(float3 v, float3 oldDir, float3 newDir) {
    return v - dot(v, newDir) * newDir;
}

public float3 computeGeomDispersion(float3 I, float3 N, float eta) {
    float3 tangent = normalize(I - dot(I, N) * N);
    if (length(tangent) < 0.0001) return float3(0);
    float3 T0, T1;
    if (!refractDir(I, N, eta, T0)) return float3(0);
    refractDir(normalize(I + tangent * 1e-4), N, eta, T1);
    return T1 - T0;
}
