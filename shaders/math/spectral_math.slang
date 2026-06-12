// Core spectral math functions
import common.types;

public float cauchyIOR(GpuMaterial m, float lamNm) {
    if (int(m.params.z) != MATERIAL_DIELECTRIC) return m.params.x;
    float lu = lamNm * 1e-3, l2 = lu * lu;
    return m.cauchyA.y + m.cauchyB.y / l2;
}

public bool refractDir(float3 I, float3 N, float eta, out float3 T) {
    float ci = -dot(I, N), st2 = eta * eta * (1.0 - ci * ci);
    if (st2 > 1.0) { T = float3(0); return false; }
    T = eta * I + (eta * ci - sqrt(1.0 - st2)) * N;
    return true;
}

public float3 reflectDir(float3 I, float3 N) { return I - 2.0 * dot(I, N) * N; }

// Evaluate a 2nd-order Legendre polynomial at normalized coordinate x ∈ [-1,1]
// f(x) = c0·P0(x) + c1·P1(x) + c2·P2(x) = c0 + c1·x + c2·(1.5x² − 0.5)
public float evalLegendre(float c0, float c1, float c2, float x) {
    return c0 + c1 * x + c2 * (3.0 * x * x - 1.0) * 0.5;
}

// Log-space: coefficients represent L(x)=ln(D(x)), return linear D(x)=exp(L(x))
public float evalDecay(float c0, float c1, float c2, float x) {
    return exp(c0 + c1 * x + c2 * (3.0 * x * x - 1.0) * 0.5);
}

// Termination: linear throughput at λ_mid (x=0) below 0.005
public bool decayIsDead(float c0, float c2) { return (c0 - 0.5 * c2) < -5.298317; }  // ln(0.005)
