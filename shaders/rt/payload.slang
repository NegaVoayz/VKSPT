// RT pipeline Payload + PushConstants shared across all RT shader modules
import common.types;
import common.bindings;
import math.sampling;

struct Payload {
    float3 color;
    float  hitT;
    float3 normal;
    int    lamStart, lamEnd;
    float3 D;        // dispersion offset vector
    float3 V;        // dispersion velocity (per-unit-distance growth)
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
    p.D = float3(0); p.V = float3(0);
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

// Generalized n-way spectral split per SplitMethodDesign.md.
// D is the dispersion offset, V the per-unit-distance growth rate.
// Split count: n = 2^k where k = ceil(log2(2|D_perp| / forceSplitWidth)).
// Legendre coefficients reprojected via closed-form shift+scale (S, Q progression).
// Wavelength intervals allocated with Bresenham integer partition.
void traceSpectralSplit(inout Payload p,
                         float3 childOrigin, float3 baseDir,
                         float3 splitD, float3 splitV,
                         float parentC0, float parentC1, float parentC2,
                         int childBounce, int childGen)
{
    // d = 2 * |D| projected onto plane perpendicular to baseDir
    float3 Dperp = splitD - dot(splitD, baseDir) * baseDir;
    float d = 2.0 * length(Dperp);

    if (d <= params.forceSplitWidth || params.forceSplitWidth <= 0.0)
        return;

    int k = int(ceil(log2(d / params.forceSplitWidth)));

    // Cap k by wavelength range: each child must have >= minSplitNm
    int lamRange = p.lamEnd - p.lamStart;
    if (params.minSplitNm > 0.0 && lamRange > 0) {
        int maxKWl = int(floor(log2(float(lamRange) / params.minSplitNm)));
        k = min(k, max(maxKWl, 0));
    }
    int n = 1 << k;

    float s  = exp2(float(-k));
    float s2 = exp2(float(-2 * k));

    float3 childD = s * splitD;
    float3 childV = s * splitV;

    int L = p.lamEnd - p.lamStart;
    int acc = 0;
    int lamLeft = p.lamStart;

    float S     = -(1.0 - s);
    float Delta = 2.0 * s;

    for (int i = 0; i < n; i++) {
        acc += L;
        int lamRightExact = p.lamStart + (acc >> k);
        int intervalW = lamRightExact - lamLeft;

        // Jitter S and wavelength boundary to break discrete banding
        float jitterS = (rand1(p.rng) - 0.5) * Delta * 0.4;
        float Sj = S + jitterS;
        float Qj = Sj * Sj;

        int maxWLJitter = max(1, intervalW / 4);
        int wlJitter = int(rand1(p.rng) * float(2 * maxWLJitter + 1)) - maxWLJitter;
        int lamRight = clamp(lamRightExact + wlJitter, lamLeft + 1, p.lamEnd);

        float3 childPos = childOrigin;
        float3 childDir = normalize(baseDir + Sj * splitV);

        float childC2 = s2 * parentC2;
        float childC1 = s * (parentC1 + 3.0 * Sj * parentC2);
        float childC0 = parentC0 + Sj * parentC1
                      + (1.5 * Qj - 0.5 + 0.5 * s2) * parentC2;

        Payload childP;
        initPayload(childP, lamLeft, lamRight, childC0, childC1, childC2,
                     childBounce, childGen, p.rng);
        childP.D = childD - dot(childD, childDir) * childDir;
        childP.V = childV - dot(childV, childDir) * childDir;

        RayDesc cr;
        cr.Origin = childPos; cr.Direction = childDir;
        cr.TMin = 0.001; cr.TMax = 10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, cr, childP);
        mergeChild(p, childP, float3(1.0));

        S += Delta;
        lamLeft = lamRight;
    }
}
