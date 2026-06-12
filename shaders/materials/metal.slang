// Metal material handler — spectral reflectance, log-space decay
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

    // L_new(x) = L_old(x) + ln(R(λ(x))) — fit via 3-point GL quadrature
    float newL[3];
    for (int k = 0; k < 3; k++) {
        float xk = (k == 0) ? -SQRT_3_5 : ((k == 1) ? 0.0 : SQRT_3_5);
        float Lk = bs.throughput + bs.decayC1 * xk + bs.decayC2 * (1.5 * xk * xk - 0.5);
        float lam_k = lamMid + halfW * xk;
        float xFull = (lam_k - LAM_CENTER) / LAM_HALF;
        float Rk = evalLegendre(r0, r1, r2, xFull);
        newL[k] = Lk + log(max(Rk, 1e-10));
    }
    bs.throughput = 0.5 * (W0*newL[0] + W1*newL[1] + W2*newL[2]);
    bs.decayC1    = 1.5 * (W0*newL[0]*(-SQRT_3_5) + W2*newL[2]*SQRT_3_5);
    float p2_0 = 0.4, p2_1 = -0.5, p2_2 = 0.4;
    bs.decayC2    = 2.5 * (W0*newL[0]*p2_0 + W1*newL[1]*p2_1 + W2*newL[2]*p2_2);

    float3 reflDir = reflectDir(bs.rayD, nrm);
    bs.rayD       = reflDir;
    bs.rayO       = pos + nrm * 0.001 + reflDir * 0.01;
    bs.lastSplit  = bs.lastSplit - 2.0 * dot(bs.lastSplit, nrm) * nrm;
    bs.disp       = bs.disp - 2.0 * dot(bs.disp, nrm) * nrm;
    bs.lastGeom   = bs.lastGeom - 2.0 * dot(bs.lastGeom, nrm) * nrm;
    bs.geomD      = bs.geomD - 2.0 * dot(bs.geomD, nrm) * nrm;
}
