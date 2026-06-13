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
    float lamMid = float(p.lamStart + p.lamEnd) * 0.5;
    float halfW  = float(p.lamEnd - p.lamStart) * 0.5;

    // Angular spread at infinity from accumulated direction dispersion
    float3 Vperp = p.V - dot(p.V, dir) * dir;
    float d = 2.0 * length(Vperp);

    // N scales with spread — cheaper than dielectric split (just envMap lookups)
    // Use a much lower threshold: 0.005 rad (~0.3°)
    float skyThreshold = 0.005;
    int k = clamp(int(ceil(log2(max(d, skyThreshold) / skyThreshold))), 0, 3);
    int N = 1 << k;
    float invN = 1.0 / float(N);
    float subHalfW = halfW * invN;

    if (!decayIsDead(p.throughput, p.decayC2) &&
        (p.throughput - 0.5 * p.decayC2) > -3.0) {

        float3 spreadDir = (d > 0.0) ? normalize(Vperp) : float3(1,0,0);
        float halfSpread = d * 0.5;

        for (int i = 0; i < N; i++) {
            float t = (float(i) + 0.5) * invN * 2.0 - 1.0;
            float3 sampleDir = normalize(dir + t * halfSpread * spreadDir);

            float2 uv;
            uv.x = atan2(sampleDir.z, sampleDir.x) * 0.159155 + 0.5;
            uv.y = acos(clamp(sampleDir.y, -1.0, 1.0)) * 0.31831;
            float3 env_t = envMap.SampleLevel(envSampler, uv, 0).xyz;

            float lam_t = lamMid + t * halfW;
            float decay_t = evalDecay(p.throughput, p.decayC1, p.decayC2, t);

            accumulateSpectrumSky(p.spectral, lam_t, subHalfW,
                                  log(max(decay_t, 1e-10)), 0.0, 0.0, env_t);
        }
    }

    p.hitT = 1e6; p.normal = float3(0,0,1);
}
