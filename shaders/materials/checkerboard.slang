// Checkerboard material handler — diffuse scattering with spectral absorption
import common.types;
import math.sampling;

[ForceInline]
public void handleCheckerboard(
    inout BounceState bs,
    inout uint rng,
    inout int stackTop,
    inout RayState rayStack[SPECTRAL_STACK_MAX],
    float3 pos, float3 nrm, float3 albedo, float3 directLin,
    int bounce, int scatSamples)
{
    float preAbsTP = bs.throughput;
    // albedo is (0.75,0.75,0.75) white or (0.15,0.15,0.15) black
    // ln(0.07+0.93*0.75) = −0.264692, ln(0.07+0.93*0.15) = −1.563156
    bs.throughput += (albedo.x > 0.5) ? -0.264692 : -1.563156;
    bs.colorAccum += directLin * exp(preAbsTP - 0.5 * bs.decayC2);

    // RR moved to bounce-start in raytrace.slang (unified with termination)

    int scatCount = int(scatSamples);
    if (scatCount < 1) scatCount = 1;
    if (scatCount > 4) scatCount = 4;
    static const float LOG_SCAT[5] = {0.0, 0.0, -0.693147, -1.098612, -1.386294};
    float scatLogTP = bs.throughput + LOG_SCAT[scatCount];
    float scatC1 = bs.decayC1;
    float scatC2 = bs.decayC2;
    for (int s = 0; s < scatCount - 1; s++) {
        if (stackTop < SPECTRAL_STACK_MAX) {
            float3 sDir = cosineSampleHemisphere(nrm, rng);
            RayState scat;
            scat.origin      = pos + nrm * 0.001 + sDir * 0.01;
            scat.direction   = sDir;
            scat.last_split  = float3(0.0); scat.dispersion  = float3(0.0);
            scat.last_geom   = float3(0.0); scat.geomDisp    = float3(0.0);
            scat.lam_start   = bs.lamStart;  scat.lam_end     = bs.lamEnd;
            scat.throughput  = scatLogTP;
            scat.decay_c1    = scatC1;     scat.decay_c2    = scatC2;
            scat.bounce      = bounce + 1; scat.generation  = bs.generation;
            rayStack[stackTop++] = scat;
        }
    }
    bs.rayD       = cosineSampleHemisphere(nrm, rng);
    bs.rayO       = pos + nrm * 0.001 + bs.rayD * 0.01;
    bs.throughput = scatLogTP; bs.decayC1 = scatC1; bs.decayC2 = scatC2;
    bs.lastSplit  = float3(0.0); bs.disp = float3(0.0);
    bs.lastGeom   = float3(0.0); bs.geomD = float3(0.0);
}
