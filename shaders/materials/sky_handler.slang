// Sky/miss handler — adaptive spectral splitting at infinity + envmap accumulation
import common.types;
import common.cmf_data;
import common.bindings;
import math.legendre_decay;
import math.color;
import math.spectral_math;

void accumulateSpectrumSky(inout float spectral[NUM_SPECTRAL_BUCKETS],
                         float lamMid, float halfWidth,
                         float c0, float c1, float c2,
                         float3 envLinear) {
    float lamS = lamMid - halfWidth;
    float lamE = lamMid + halfWidth;
    float bandWidth = halfWidth * 2.0;

    // Flat decay (c1≈c2≈0): D(λ)=exp(c0) everywhere — single exp for all buckets
    if (abs(c1) < 0.001 && abs(c2) < 0.001) {
        float D_flat = exp(c0);
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
            float bS = float(LAMBDA_MIN + 40 * b);
            float bE = bS + 40.0;
            float oS = max(lamS, bS);
            float oE = min(lamE, bE);
            if (oS >= oE) continue;
            float lamEval = (oS + oE) * 0.5;
            float S_ev  = estimateSpectrum(lamEval, envLinear);
            float bw    = oE - oS;
            spectral[b] += S_ev * D_flat * bw;
        }
        return;
    }

    // Narrow band: D(λ) ≈ D(λ_mid) across the band — single exp
    if (bandWidth < 120.0) {
        float D_mid = exp(c0 - 0.5 * c2);
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
            float bS = float(LAMBDA_MIN + 40 * b);
            float bE = bS + 40.0;
            float oS = max(lamS, bS);
            float oE = min(lamE, bE);
            if (oS >= oE) continue;
            float lamEval = (oS + oE) * 0.5;
            float S_ev  = estimateSpectrum(lamEval, envLinear);
            float bw    = oE - oS;
            spectral[b] += S_ev * D_mid * bw;
        }
        return;
    }

    // Wide band + non-flat decay: per-bucket evaluation (rare)
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
        float bS = float(LAMBDA_MIN + 40 * b);
        float bE = bS + 40.0;
        float oS = max(lamS, bS);
        float oE = min(lamE, bE);
        if (oS >= oE) continue;
        float lamEval = (oS + oE) * 0.5;
        float xEv   = clamp((lamEval - lamMid) / max(halfWidth, 0.001), -1.0, 1.0);
        float D_ev  = evalDecay(c0, c1, c2, xEv);
        float S_ev  = estimateSpectrum(lamEval, envLinear);
        float bw    = oE - oS;
        spectral[b] += S_ev * D_ev * bw;
    }
}

[ForceInline]
public void handleSkyMiss(
    inout BounceState bs,
    inout int stackTop,
    inout RayState rayStack[SPECTRAL_STACK_MAX],
    inout float spectral[NUM_SPECTRAL_BUCKETS],
    float3 rayO, float3 rayD,
    int bounce, int sp, int2 px, int2 sz,
    float fovTan, float splitMult)
{
    // Note: caller handles G-buffer write for sp==0 && bounce==0 already

    float estSkyDist = 10000.0;
    float3  skyDisp    = bs.lastSplit + bs.disp * estSkyDist;
    float skySepMag  = length(skyDisp);
    float skyPixHW   = fovTan / float(sz.x);
    float skyThresh  = estSkyDist * skyPixHW * 0.5 * splitMult;

    int lamRange = bs.lamEnd - bs.lamStart;
    if (skySepMag > skyThresh && lamRange >= 20 && !decayIsDead(bs.throughput, bs.decayC2) && bs.generation < 7) {
        int midWl = (bs.lamStart + bs.lamEnd) / 2;
        float3 skyPerp = normalize(skyDisp);
        if (length(skyPerp) < 0.0001) skyPerp = float3(1,0,0);

        if (stackTop < SPECTRAL_STACK_MAX - 1) {
            RayState skyV;
            skyV.direction   = normalize(rayD + skyPerp * skySepMag * 0.5 / estSkyDist);
            skyV.origin      = rayO;
            skyV.last_split  = bs.lastSplit;
            skyV.dispersion  = bs.disp;
            skyV.lam_start   = bs.lamStart;
            skyV.lam_end     = midWl;
            reprojectDecayLeft(bs.throughput, bs.decayC1, bs.decayC2,
                               skyV.throughput, skyV.decay_c1, skyV.decay_c2);
            skyV.bounce      = bounce;
            skyV.generation  = bs.generation + 1;
            rayStack[stackTop++] = skyV;

            RayState skyR;
            skyR.direction   = normalize(rayD - skyPerp * skySepMag * 0.5 / estSkyDist);
            skyR.origin      = rayO;
            skyR.last_split  = bs.lastSplit;
            skyR.dispersion  = bs.disp;
            skyR.lam_start   = midWl;
            skyR.lam_end     = bs.lamEnd;
            reprojectDecayRight(bs.throughput, bs.decayC1, bs.decayC2,
                                skyR.throughput, skyR.decay_c1, skyR.decay_c2);
            skyR.bounce      = bounce;
            skyR.generation  = bs.generation + 1;
            rayStack[stackTop++] = skyR;
            return;
        }
    }

    float lamMid  = float(bs.lamStart + bs.lamEnd) * 0.5;
    float halfW   = float(bs.lamEnd - bs.lamStart) * 0.5;
    float2 uv;
    uv.x = atan2(rayD.z, rayD.x) * 0.159155 + 0.5;
    uv.y = acos(clamp(rayD.y, -1.0, 1.0)) * 0.31831;
    float3 envLinear = envMap.SampleLevel(envSampler, uv, 0).xyz;
    accumulateSpectrumSky(spectral, lamMid, halfW,
                        bs.throughput, bs.decayC1, bs.decayC2, envLinear);
}
