// RT pipeline sky handler — spectrum accumulation + miss-shader split logic
import common.types;
import common.cmf_data;
import common.bindings;
import math.color;
import math.spectral_math;
import rt.payload;

void accumulateSpectrumSky(inout float spectral[NUM_SPECTRAL_BUCKETS],
                           float lamMid, float halfW,
                           float c0, float c1, float c2, float3 envLinear)
{
    float lamS = lamMid - halfW, lamE = lamMid + halfW;
    float bw = halfW * 2.0;
    if (abs(c1) < 0.001 && abs(c2) < 0.001) {
        float D = exp(c0);
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
            float bS = float(LAMBDA_MIN + 40*b), bE = bS + 40.0;
            float oS = max(lamS, bS), oE = min(lamE, bE);
            if (oS >= oE) continue;
            spectral[b] += estimateSpectrum((oS+oE)*0.5, envLinear) * D * (oE-oS);
        }
        return;
    }
    if (bw < 120.0) {
        float D = exp(c0 - 0.5*c2);
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
            float bS = float(LAMBDA_MIN + 40*b), bE = bS + 40.0;
            float oS = max(lamS, bS), oE = min(lamE, bE);
            if (oS >= oE) continue;
            spectral[b] += estimateSpectrum((oS+oE)*0.5, envLinear) * D * (oE-oS);
        }
        return;
    }
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
        float bS = float(LAMBDA_MIN + 40*b), bE = bS + 40.0;
        float oS = max(lamS, bS), oE = min(lamE, bE);
        if (oS >= oE) continue;
        float xE = clamp(((oS+oE)*0.5 - lamMid) / max(halfW, 0.001), -1.0, 1.0);
        float D  = evalDecay(c0, c1, c2, xE);
        spectral[b] += estimateSpectrum((oS+oE)*0.5, envLinear) * D * (oE-oS);
    }
}

void handleSkyMiss(inout Payload p)
{
    if (p.bounce < 0) { p.isShadowed = 1.0; return; }

    float3 dir = WorldRayDirection();
    float3 skyDisp = p.lastSplit + p.disp * 10000.0;
    float skySepMag = length(skyDisp);
    int lamRange = p.lamEnd - p.lamStart;
    int minRange = max(int(params.minSplitNm), 20);

    if (skySepMag > params.forceSplitWidth && params.forceSplitWidth > 0.0 &&
        lamRange >= minRange &&
        !decayIsDead(p.throughput, p.decayC2) && p.generation < 7) {

        traceSpectralSplit(p, WorldRayOrigin(), dir,
                           normalize(skyDisp), skySepMag, 10000.0,
                           p.throughput, p.decayC1, p.decayC2,
                           p.bounce, p.generation+1,
                           p.lastSplit, p.disp);
        p.hitT = 1e6; p.normal = float3(0,0,1);
        return;
    }

    float lamMid = float(p.lamStart+p.lamEnd)*0.5;
    float halfW  = float(p.lamEnd-p.lamStart)*0.5;
    float2 uv;
    uv.x = atan2(dir.z, dir.x)*0.159155+0.5;
    uv.y = acos(clamp(dir.y, -1.0, 1.0))*0.31831;
    float3 envLinear = envMap.SampleLevel(envSampler, uv, 0).xyz;
    accumulateSpectrumSky(p.spectral, lamMid, halfW,
                          p.throughput, p.decayC1, p.decayC2, envLinear);
    p.hitT = 1e6; p.normal = float3(0,0,1);
}
