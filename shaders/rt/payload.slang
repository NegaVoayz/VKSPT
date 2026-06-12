// RT pipeline Payload + PushConstants shared across all RT shader modules
import common.types;
import common.bindings;
import math.legendre_decay;

struct Payload {
    float3 color;
    float  hitT;
    float3 normal;
    int    lamStart, lamEnd;
    float3 lastSplit;
    float3 disp;
    float  throughput;
    float  decayC1, decayC2;
    int    bounce, generation;
    uint   rng;
    float  spectral[NUM_SPECTRAL_BUCKETS];
    float  isShadowed;
};

struct PushConstants {
    float3 camOrigin;   float _pad0;
    float3 camU;        float _pad1;
    float3 camV;        float _pad2;
    float3 camW;        float _pad3;
    int   samplesPerPixel, maxBounces, materialCount;
    float fovTan, splitMult, forceSplitWidth;
    int   scatterSamples; float mergeThreshold;
    int   frameIndex;
    float diffuseStrength; float specularStrength; int numLights;
    float minSplitNm;
};
[[vk::push_constant]] PushConstants params;

void initPayload(out Payload p, int ls, int le, float tp, float c1, float c2,
                 int bounce, int gen, uint rng)
{
    p.color = float3(0.0); p.normal = float3(0,0,1);
    p.lamStart = ls; p.lamEnd = le;
    p.lastSplit = float3(0); p.disp = float3(0);
    p.throughput = tp; p.decayC1 = c1; p.decayC2 = c2;
    p.bounce = bounce; p.generation = gen; p.rng = rng;
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) p.spectral[b] = 0.0;
}

void mergeChild(inout Payload parent, Payload child, float3 weight)
{
    parent.color += weight * child.color;
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        parent.spectral[b] += child.spectral[b];
}

// Binary spectral split: creates two child rays offset perpendicular to baseDir,
// reprojects decay coefficients for each half-wavelength sub-interval.
// Caller is responsible for checking split conditions (mag > forceSplitWidth, etc.)
void traceSpectralSplit(inout Payload p,
                         float3 childOrigin, float3 baseDir,
                         float3 splitPerp, float splitMag, float estDist,
                         float parentTp, float parentC1, float parentC2,
                         int childBounce, int childGen,
                         float3 childSplit, float3 childDisp)
{
    int midWl = (p.lamStart + p.lamEnd)/2;
    float offsetScale = splitMag * 0.5 / estDist;
    float3 perp = splitPerp;
    if (length(perp) < 0.0001) perp = float3(1,0,0);

    {
        float3 vdir = normalize(baseDir + perp * offsetScale);
        Payload vp; initPayload(vp, p.lamStart, midWl,
            0,0,0, childBounce, childGen, p.rng);
        reprojectDecayLeft(parentTp, parentC1, parentC2,
                           vp.throughput, vp.decayC1, vp.decayC2);
        vp.lastSplit = childSplit; vp.disp = childDisp;
        RayDesc vr; vr.Origin = childOrigin; vr.Direction = vdir;
        vr.TMin = 0.001; vr.TMax = 10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, vr, vp);
        mergeChild(p, vp, float3(1.0));
    }
    {
        float3 rdir = normalize(baseDir - perp * offsetScale);
        Payload rp; initPayload(rp, midWl, p.lamEnd,
            0,0,0, childBounce, childGen, p.rng);
        reprojectDecayRight(parentTp, parentC1, parentC2,
                            rp.throughput, rp.decayC1, rp.decayC2);
        rp.lastSplit = childSplit; rp.disp = childDisp;
        RayDesc rr; rr.Origin = childOrigin; rr.Direction = rdir;
        rr.TMin = 0.001; rr.TMax = 10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
        mergeChild(p, rp, float3(1.0));
    }
}
