// Metal material handler — spectral reflectance via Gauss-Legendre quadrature
import common.types;
import math.spectral_math;

[ForceInline]
public void handleMetal(
    inout BounceState bs,
    float3 pos, float3 nrm, GpuMaterial mat)
{
    float3 ma = mat.albedo.xyz;
    float r0 = ma.y + (ma.x + ma.z - 2.0*ma.y) / 6.0;
    float r1 = (ma.x - ma.z) * 0.5;
    float r2 = (ma.x + ma.z - 2.0*ma.y) / 3.0;
    const float SQRT_3_5 = 0.77459667;
    const float W0 = 5.0/9.0, W1 = 8.0/9.0, W2 = 5.0/9.0;
    float lamMid = float(bs.lamStart + bs.lamEnd) * 0.5;
    float halfW  = float(bs.lamEnd - bs.lamStart) * 0.5;
    const float LAM_CENTER = float(LAMBDA_MIN + LAMBDA_MAX) * 0.5;
    const float LAM_HALF   = float(LAMBDA_MAX - LAMBDA_MIN) * 0.5;
    float newD[3];
    for (int k = 0; k < 3; k++) {
        float xk = (k == 0) ? -SQRT_3_5 : ((k == 1) ? 0.0 : SQRT_3_5);
        float Dk = evalDecay(bs.throughput, bs.decayC1, bs.decayC2, xk);
        float lam_k = lamMid + halfW * xk;
        float xFull = (lam_k - LAM_CENTER) / LAM_HALF;
        float Rk = evalDecay(r0, r1, r2, xFull);
        newD[k] = Dk * Rk;
    }
    bs.throughput = 0.5 * (W0*newD[0] + W1*newD[1] + W2*newD[2]);
    bs.decayC1    = 1.5 * (W0*newD[0]*(-SQRT_3_5) + W2*newD[2]*SQRT_3_5);
    float p2_0 = 0.4, p2_1 = -0.5, p2_2 = 0.4;
    bs.decayC2    = 2.5 * (W0*newD[0]*p2_0 + W1*newD[1]*p2_1 + W2*newD[2]*p2_2);
    float3 reflDir = reflectDir(bs.rayD, nrm);
    bs.rayD       = reflDir;
    bs.rayO       = pos + nrm * 0.001 + reflDir * 0.01;
    bs.lastSplit  = bs.lastSplit - 2.0 * dot(bs.lastSplit, nrm) * nrm;
    bs.disp       = bs.disp - 2.0 * dot(bs.disp, nrm) * nrm;
    bs.lastGeom   = bs.lastGeom - 2.0 * dot(bs.lastGeom, nrm) * nrm;
    bs.geomD      = bs.geomD - 2.0 * dot(bs.geomD, nrm) * nrm;
}
