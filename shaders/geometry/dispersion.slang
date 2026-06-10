// Dispersion & split utilities

vec3 computeDispersion(vec3 I, vec3 N, float eta_mid, float deta) {
    vec3 T0, T1;
    float h = deta * 1e-3;
    if (abs(h) < 1e-9) return vec3(0);
    if (!refractDir(I, N, eta_mid, T0)) return vec3(0);
    if (!refractDir(I, N, eta_mid + h, T1)) return vec3(0);
    vec3 dT_deta = (T1 - T0) / h;
    vec3 spread = dT_deta * deta;
    return spread - dot(spread, T0) * T0;
}

vec3 rotateToNewFrame(vec3 v, vec3 oldDir, vec3 newDir) {
    return v - dot(v, newDir) * newDir;
}

vec3 computeGeomDispersion(vec3 I, vec3 N, float eta) {
    vec3 tangent = normalize(I - dot(I, N) * N);
    if (length(tangent) < 0.0001) return vec3(0);
    vec3 T0, T1;
    if (!refractDir(I, N, eta, T0)) return vec3(0);
    refractDir(normalize(I + tangent * 1e-4), N, eta, T1);
    return T1 - T0;
}
