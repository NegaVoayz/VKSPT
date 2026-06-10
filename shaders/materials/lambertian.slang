// Lambertian material handler — spectral absorption + diffuse scattering
import common.types;
import math.color;
import math.sampling;

[ForceInline]
public void handleLambertian(
    inout BounceState bs,
    inout uint rng,
    inout int stackTop,
    inout RayState rayStack[SPECTRAL_STACK_MAX],
    float3 pos, float3 nrm, GpuMaterial mat, float3 directLin,
    int bounce, int scatSamples)
{
    float nl = rgbToDomLambda(mat.albedo.xyz);
    float lum = dot(mat.albedo.xyz, float3(0.299, 0.587, 0.114));
    float maxC = max(max(mat.albedo.xyz.x, mat.albedo.xyz.y), mat.albedo.xyz.z);
    float minC = min(min(mat.albedo.xyz.x, mat.albedo.xyz.y), mat.albedo.xyz.z);
    float sat = (maxC - minC) / max(maxC, 0.001);

    float lamS = float(bs.lamStart), lamE = float(bs.lamEnd);
    float lamM = (lamS + lamE) * 0.5;
    float dS = lamS - nl, dM = lamM - nl, dE = lamE - nl;
    float attAvg = (exp(-0.5*dS*dS/7200.0) + exp(-0.5*dM*dM/7200.0) + exp(-0.5*dE*dE/7200.0)) / 3.0;

    float preAbsTP = bs.throughput;
    float attMix = lerp(1.0, attAvg, sat);
    bs.throughput *= attMix * (0.07 + 0.93 * lum);

    bs.colorAccum += directLin * preAbsTP;

    if (bounce >= 3) {
        float survive = max(bs.throughput, 0.1);
        if (rand1(rng) > survive) { bs.throughput = 0.0; return; }
        bs.throughput /= survive;
    }

    int scatCount = int(scatSamples);
    if (scatCount < 1) scatCount = 1;
    if (scatCount > 4) scatCount = 4;
    float scatThroughput = bs.throughput / float(scatCount);
    float scatC1 = bs.decayC1 / float(scatCount);
    float scatC2 = bs.decayC2 / float(scatCount);
    for (int s = 0; s < scatCount - 1; s++) {
        if (stackTop < SPECTRAL_STACK_MAX) {
            float3 sDir = cosineSampleHemisphere(nrm, rng);
            RayState scat;
            scat.origin      = pos + nrm * 0.001 + sDir * 0.01;
            scat.direction   = sDir;
            scat.last_split  = float3(0.0);
            scat.dispersion  = float3(0.0);
            scat.last_geom   = float3(0.0);
            scat.geomDisp    = float3(0.0);
            scat.lam_start   = bs.lamStart;
            scat.lam_end     = bs.lamEnd;
            scat.throughput  = scatThroughput;
            scat.decay_c1    = scatC1;
            scat.decay_c2    = scatC2;
            scat.bounce      = bounce + 1;
            scat.generation  = bs.generation;
            rayStack[stackTop++] = scat;
        }
    }
    bs.rayD       = cosineSampleHemisphere(nrm, rng);
    bs.rayO       = pos + nrm * 0.001 + bs.rayD * 0.01;
    bs.throughput = scatThroughput;
    bs.decayC1    = scatC1;
    bs.decayC2    = scatC2;
    bs.lastSplit  = float3(0.0); bs.disp = float3(0.0);
    bs.lastGeom   = float3(0.0); bs.geomD = float3(0.0);
}
