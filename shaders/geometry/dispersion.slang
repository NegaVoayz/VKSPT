// Dispersion & split utilities
import math.spectral_math;

public float3 computeDispersion(float3 I, float3 N, float eta_mid, float deta) {
    float3 t0, t1;
    float h = deta * 1e-3;
    if (abs(h) < 1e-9) return float3(0);
    if (!refractDir(I, N, eta_mid, t0)) return float3(0);
    if (!refractDir(I, N, eta_mid + h, t1)) return float3(0);
    float3 dT_deta = (t1 - t0) / h;
    float3 spread = dT_deta * deta;
    return spread - dot(spread, t0) * t0;
}

public float3 rotateToNewFrame(float3 v, float3 oldDir, float3 newDir) {
    return v - dot(v, newDir) * newDir;
}

public float3 computeGeomDispersion(float3 I, float3 N, float eta) {
    float3 tangent = normalize(I - dot(I, N) * N);
    if (length(tangent) < 0.0001) return float3(0);
    float3 t0, t1;
    if (!refractDir(I, N, eta, t0)) return float3(0);
    refractDir(normalize(I + tangent * 1e-4), N, eta, t1);
    return t1 - t0;
}
