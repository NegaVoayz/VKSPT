// Full spectral RT pipeline — Legendre log-decay + dispersion + sky splitting
// Entry points only; material/sky/light logic lives in shaders/rt/
import common.types;
import common.cmf_data;
import common.bindings;
import geometry.normals;
import math.sampling;
import math.spectral_math;
import math.color;
import rt.payload;
import rt.direct_light;
import rt.sky;
import rt.material_handlers;
import photon.photon_light;

[shader("raygeneration")]
void RayGenMain()
{
    uint2 px = DispatchRaysIndex().xy;
    uint2 sz = DispatchRaysDimensions().xy;
    if (px.x >= sz.x || px.y >= sz.y) return;

    float2 uv = (float2(px) + 0.5) / float2(sz) * 2.0 - 1.0;
    uv.y = -uv.y;
    float3 camDir = normalize(params.camW + uv.x*params.camU + uv.y*params.camV);
    float3 origin = params.camOrigin;
    int spp = max(1, params.samplesPerPixel);

    float3 pixelColor = float3(0.0);
    float3 gbufNormal = float3(0.0);
    float  gbufDepth  = 1e6;

    for (int sp = 0; sp < spp; sp++) {
        Payload p;
        uint seed = wangHash(uint(px.x)*1973u + uint(px.y)*9277u +
                             uint(sp)*2654435761u +
                             uint(params.frameIndex)*134775813u);
        initPayload(p, LAMBDA_MIN, LAMBDA_MAX, 0.0, 0.0, 0.0, 0, 0, 0, float3(0,0,0), seed);

        RayDesc ray;
        ray.Origin = origin; ray.Direction = camDir;
        ray.TMin = 0.001; ray.TMax = 10000.0;
        incPrimaryRay(0);
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, ray, p);

        if (sp == 0) { gbufNormal = p.normal; gbufDepth = p.hitT; }

        pixelColor += p.color;
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            pixelColor += (p.spectral[b] / 40.0) * BUCKET_CMF[b];
    }

    normalImage[int2(px)] = float4(gbufNormal*0.5+0.5, 1.0);
    depthImage[int2(px)]  = gbufDepth;

    uint w, h; outputImage.GetDimensions(w, h);
    uint base = (uint(px.y)*w + uint(px.x))*4u;
    float3 oldAcc = float3(accumData[base], accumData[base+1], accumData[base+2]);
    float oldCnt = accumData[base+3];
    float3 newAcc = oldAcc + pixelColor;
    float newCnt = oldCnt + float(spp);
    accumData[base]=newAcc.x; accumData[base+1]=newAcc.y;
    accumData[base+2]=newAcc.z; accumData[base+3]=newCnt;
    outputImage[int2(px)] = float4(linearToSRGB(newAcc/max(newCnt,1.0)*2.5), 1.0);
}

// ---- Photon tracing raygen ----
[shader("raygeneration")]
void PhotonRayGenMain()
{
    uint photonIdx = DispatchRaysIndex().x;
    uint batchPhotons = uint(max(1, params.photonCount));

    if (photonIdx >= batchPhotons) return;

    // Each batch is dedicated to one light source.
    // Light config comes from push constants (bypasses lights buffer for reliability).
    int li = clamp(params.batchIndex, 0, max(0, params.numLights - 1));
    GpuLight l = lights.items[li];
    float3 lightRGB = float3(params.lightColorPC[0], params.lightColorPC[1], params.lightColorPC[2]);
    float lightIntensity = params.lightIntensityPC;

    uint seed = wangHash(photonIdx ^ (uint(params.frameIndex) + uint(li) * 31337u) * 134775813u);

    // Sample position and direction on the selected light
    float3 origin, direction;
    sampleLightSource(l, origin, direction, seed);
    
    float c0 = log(max(lightIntensity, 1e-10) / float(max(batchPhotons, 1u)));
    float c1 = 0.0;
    float c2 = 0.0;

    Payload p;
    initPayload(p, LAMBDA_MIN, LAMBDA_MAX, c0, c1, c2,
                0, 0, 0, lightRGB, seed);
    p.specularChain = 0;

    RayDesc ray;
    ray.Origin = origin; ray.Direction = direction;
    ray.TMin = 0.001; ray.TMax = 10000.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, ray, p);
}

[shader("closesthit")]
void ClosestHitMain(inout Payload p, in BuiltInTriangleIntersectionAttributes attr)
{
    p.hitT = RayTCurrent();
    float3 pos = WorldRayOrigin() + WorldRayDirection() * p.hitT;
    float3 rayDir = WorldRayDirection();
    int mi = InstanceIndex(), pid = PrimitiveIndex();

    float3 nrm;
    if (instanceRanges[0].useSmoothNormals[mi] != 0u)
        nrm = computeSmoothNormal(mi, pid, attr.barycentrics);
    else nrm = computeGeometricNormal(mi, pid);
    float3x3 Nm = float3x3(instanceNormals[0].normalCol0[mi].xyz,
                         instanceNormals[0].normalCol1[mi].xyz,
                         instanceNormals[0].normalCol2[mi].xyz);
    nrm = normalize(mul(nrm, Nm));
    float3 fNrm = dot(nrm, rayDir) < 0.0 ? nrm : -nrm;
    p.normal = fNrm;

    uint matIdx = instanceRanges[0].materialID[mi];
    GpuMaterial mat = materials.items[clamp(matIdx, 0u,
        max(0u, uint(params.materialCount)-1u))];
    int matType = int(mat.params.z);

    // ---- RR + bounce limit (both passes) ----
    bool isDead = decayIsDead(p.throughput, p.decayC2);
    if (params.passType != PASS_PHOTON && isDead)
        { p.color = float3(0.0); return; }

    // ---- Split check (before apply_diff) ----
    // D,V are ⟂ rayDir from previous bounce; check unpropagated spread.
    // Split is purely geometric: spread > threshold means different
    // wavelength bands are at noticeably different positions and may hit
    // different geometry. The parent dies — children re-hit the surface
    // at dispersed positions, then go through FULL material handling
    // (Fresnel MC, reflect/refract/diffuse) independently.
    float3 Dperp = p.D - dot(p.D, rayDir) * rayDir;
    float splitSpread = 2.0 * length(Dperp);
    bool splitOK = splitSpread > params.forceSplitWidth
                && params.forceSplitWidth > 0.0
                && p.generation < 7 && !isDead;

    if (splitOK) {
        traceSpectralSplit(p, pos + rayDir * 0.001, rayDir,
                           p.D, p.V,
                           p.throughput, p.decayC1, p.decayC2,
                           p.bounce + 1, p.generation + 1);
        return;
    }

    // ---- apply_diff: propagate D by travel distance ----
    p.D = (p.D + p.V * p.hitT) - dot(p.D + p.V * p.hitT, rayDir) * rayDir;

    // ---- Pass dispatch ----
    if (params.passType == PASS_PHOTON) {
        handlePhotonHit(p, pos, fNrm, nrm, matType);
        return;
    }

    // Camera path below
    if (p.bounce >= 3) {
        float linTp = exp(p.throughput - 0.5 * p.decayC2);
        float survive = max(linTp, 0.1);
        if (rand1(p.rng) > survive) { p.color = float3(0.0); return; }
        p.throughput -= log(survive);
    }
    if (p.bounce >= params.maxBounces || decayIsDead(p.throughput, p.decayC2))
        { p.color = float3(0.0); return; }

    if (matType == MATERIAL_CHECKERBOARD)
        handleCheckerboard(p, pos, fNrm, p.rng);
    else if (matType == MATERIAL_LAMBERTIAN)
        handleLambertian(p, pos, fNrm, mat, p.rng);
    else if (matType == MATERIAL_METAL)
        handleMetal(p, pos, fNrm, mat, p.rng);
    else if (matType == MATERIAL_DIELECTRIC)
        handleDielectric(p, pos, nrm, mat);
    else
        p.color = float3(0.5);
}

[shader("miss")]
void MissMain(inout Payload p)
{
    if (params.passType == PASS_PHOTON) return;
    handleSkyMiss(p);
}
