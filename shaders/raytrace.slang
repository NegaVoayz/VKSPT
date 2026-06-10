import common.types;
import common.cmf_data;
import math.spectral_math;
import math.color;
import math.absorption;
import math.legendre_decay;
import math.sampling;
import geometry.dispersion;
import geometry.normals;
import materials.metal;
import materials.checkerboard;
import materials.lambertian;
import materials.sky_handler;
import materials.dielectric;

// =============================================================================
// Descriptors
// =============================================================================
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> outputImage;

struct MaterialBlock { GpuMaterial items[16]; };
[[vk::binding(2, 0)]] ConstantBuffer<MaterialBlock> materials;
struct LightBlock { GpuLight items[4]; };
[[vk::binding(3, 0)]] ConstantBuffer<LightBlock> lights;

[[vk::binding(4, 0)]]  StructuredBuffer<float> vertexData;
[[vk::binding(5, 0)]]  StructuredBuffer<uint>  indexData;
[[vk::binding(12, 0)]] StructuredBuffer<float> normalData;
[[vk::binding(13, 0)]] RWStructuredBuffer<float> accumData;

[[vk::binding(14, 0)]] RWTexture2D<float4> normalImage;
[[vk::binding(15, 0)]] RWTexture2D<float>  depthImage;

static const int MAX_INSTANCES = 16;
struct InstanceRangeBlock {
    uint vtxOffset[MAX_INSTANCES];
    uint vtxCount[MAX_INSTANCES];
    uint idxOffset[MAX_INSTANCES];
    uint idxCount[MAX_INSTANCES];
    uint materialID[MAX_INSTANCES];
    uint useSmoothNormals[MAX_INSTANCES];
};
[[vk::binding(6, 0)]] StructuredBuffer<InstanceRangeBlock> instanceRanges;

struct InstanceNormalBlock {
    float4 normalCol0[MAX_INSTANCES];
    float4 normalCol1[MAX_INSTANCES];
    float4 normalCol2[MAX_INSTANCES];
};
[[vk::binding(16, 0)]] StructuredBuffer<InstanceNormalBlock> instanceNormals;

[[vk::binding(11, 0)]] Texture2D<float4> envMap;
[[vk::binding(11, 0)]] SamplerState envSampler;

// =============================================================================
// Sky + Direct lighting + Spectral accumulation
// =============================================================================

float3 sampleDirectLighting(float3 pos, float3 nrm, float3 albedo, uint rng) {
    float3 contrib = float3(0.0);
    for (int li = 0; li < 4; li++) {
        float4 ptype = lights.items[li].pos_type;
        int ltype = int(ptype.w);
        if (ltype == LIGHT_AMBIENT) continue;
        float4 cinfo = lights.items[li].color_intensity;
        float3 lcolor = cinfo.xyz;
        float lintensity = cinfo.w;
        if (lintensity <= 0.0) continue;
        float3 toLight;
        float distToLight;
        float tMax;
        float3 lpos = ptype.xyz;
        if (ltype == LIGHT_DIRECTIONAL) {
            toLight = normalize(-lpos);
            distToLight = 1e6;
            tMax = 10000.0;
        } else {
            toLight = lpos - pos;
            distToLight = length(toLight);
            float maxDist = lights.items[li].outer_range.y;
            if (distToLight > maxDist && maxDist > 0.0) continue;
            toLight /= distToLight;
            tMax = distToLight * 0.999;
        }
        float nDotL = dot(nrm, toLight);
        if (nDotL <= 0.0) continue;
        float spotAtt = 1.0;
        if (ltype == LIGHT_SPOT) {
            float4 dinfo = lights.items[li].dir_inner;
            float3 ldir = dinfo.xyz;
            float inAng  = dinfo.w * 3.14159265 / 180.0;
            float outAng = lights.items[li].outer_range.x * 3.14159265 / 180.0;
            float cosPhi = dot(-toLight, normalize(ldir));
            float cosIn  = cos(inAng);
            float cosOut = cos(outAng);
            spotAtt = clamp((cosPhi - cosOut) / max(cosIn - cosOut, 0.001), 0.0, 1.0);
            if (spotAtt <= 0.0) continue;
        }
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        float3 shadowOrg = pos + nrm * 0.001 + toLight * 0.01;
        RayDesc shadowRay = { shadowOrg, 0.001, toLight, tMax };
        rq.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, shadowRay);
        while (rq.Proceed()) {}
        if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            float r2 = distToLight * distToLight + 1.0;
            float att = lintensity / max(r2, 1.0);
            if (ltype == LIGHT_DIRECTIONAL) att = lintensity;
            contrib += lcolor * (albedo * 0.31831) * (att * nDotL * spotAtt);
        }
    }
    return contrib;
}

// =============================================================================
// Push constants
// =============================================================================

struct PushConstants {
    float3 camOrigin;   float _pad0;
    float3 camU;        float _pad1;
    float3 camV;        float _pad2;
    float3 camW;        float _pad3;
    int   samplesPerPixel, maxBounces, materialCount;
    float fovTan, splitMult, forceSplitWidth;
    int   scatterSamples; float mergeThreshold;
    int   frameIndex; int _padEnd0; int _padEnd1;
};
[[vk::push_constant]] PushConstants params;

// =============================================================================
// Main — adaptive spectral path tracing with per-pixel ray stack
// =============================================================================

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID) {
    int2 px = int2(dispatchID.xy);
    int w, h;
    outputImage.GetDimensions(w, h);
    int2 sz = int2(w, h);
    if (px.x >= sz.x || px.y >= sz.y) return;

    float3 origin  = params.camOrigin;
    RayState rayStack[SPECTRAL_STACK_MAX];
    float3 pixelColor = float3(0.0);
    int spp = max(1, params.samplesPerPixel);

    for (int sp = 0; sp < spp; sp++) {
        uint rngBase = wangHash(uint(px.x) * 1973u + uint(px.y) * 9277u + uint(sp) * 2654435761u + uint(params.frameIndex) * 134775813u);
        float jx = (float(rngBase & 0xFFFFu) / 65536.0) - 0.5;
        float jy = (float((rngBase >> 16u) & 0xFFFFu) / 65536.0) - 0.5;
        float2 uv = (float2(px) + 0.5 + float2(jx, jy)) / float2(sz) * 2.0 - 1.0;
        uv.y = -uv.y;
        float3 camDir = normalize(params.camW + uv.x * params.camU + uv.y * params.camV);

        int stackTop = 0;
        RayState init;
        init.origin      = origin;
        init.direction   = camDir;
        init.last_split  = float3(0.0);
        init.dispersion  = float3(0.0);
        init.geomDisp   = float3(0.0);
        init.last_geom  = float3(0.0);
        init.lam_start   = LAMBDA_MIN;
        init.lam_end     = LAMBDA_MAX;
        init.throughput  = 1.0;
        init.decay_c1   = 0.0;
        init.decay_c2   = 0.0;
        init.bounce      = 0;
        init.generation  = 0;
        rayStack[stackTop++] = init;

        float spectral[NUM_SPECTRAL_BUCKETS];
        for (int sb = 0; sb < NUM_SPECTRAL_BUCKETS; sb++) spectral[sb] = 0.0;
        float3  colorAccum = float3(0.0);

    while (stackTop > 0) {
        RayState r = rayStack[--stackTop];
        uint rng = wangHash(rngBase ^ uint(stackTop + 1) * 127u);

        float3  rayO       = r.origin;
        float3  rayD       = r.direction;
        float3  lastSplit  = r.last_split;
        float3  disp       = r.dispersion;
        float3  lastGeom   = r.last_geom;
        float3  geomD      = r.geomDisp;
        float throughput = r.throughput;
        float decayC1   = r.decay_c1;
        float decayC2   = r.decay_c2;
        int   lamStart   = r.lam_start;
        int   lamEnd     = r.lam_end;
        int   generation = r.generation;

        for (int bounce = r.bounce; bounce < params.maxBounces; ++bounce) {
            if (throughput < 0.005) break;

            RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
            RayDesc ray = { rayO, 0.001, rayD, 10000.0 };
            rq.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
            while (rq.Proceed()) {}

            if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                // ---- Miss → sky ----
                if (sp == 0 && bounce == 0) {
                    normalImage[int2(px.x, px.y)] = float4(0.0, 0.0, 0.0, 1.0);
                    depthImage[int2(px.x, px.y)] = 1e6;
                }

                BounceState bs;
                bs.rayO = rayO; bs.rayD = rayD;
                bs.throughput = throughput; bs.decayC1 = decayC1; bs.decayC2 = decayC2;
                bs.lastSplit = lastSplit; bs.disp = disp;
                bs.lastGeom = lastGeom; bs.geomD = geomD;
                bs.lamStart = lamStart; bs.lamEnd = lamEnd;
                bs.generation = generation; bs.colorAccum = colorAccum;
                handleSkyMiss(bs, stackTop, rayStack, spectral,
                              rayO, rayD, bounce, sp, px, sz,
                              params.fovTan, params.splitMult);
                // Sky handler pushes split rays OR accumulates spectrum
                // Unpack modified state
                throughput = bs.throughput; decayC1 = bs.decayC1; decayC2 = bs.decayC2;
                lamStart = bs.lamStart; lamEnd = bs.lamEnd;
                generation = bs.generation; colorAccum = bs.colorAccum;
                lastSplit = bs.lastSplit; disp = bs.disp;
                lastGeom = bs.lastGeom; geomD = bs.geomD;
                break;
            }

            // ---- Hit ----
            float  tHit = rq.CommittedRayT();
            lastSplit  += disp * tHit;
            lastGeom   += geomD * tHit;
            float3   pos  = rayO + rayD * tHit;
            int    mi   = rq.CommittedInstanceID();
            int primitiveID = rq.CommittedPrimitiveIndex();
            float3 nrm;
            if (instanceRanges[0].useSmoothNormals[mi] != 0u) {
                float2 bary = rq.CommittedTriangleBarycentrics();
                nrm = computeSmoothNormal(mi, primitiveID, bary);
            } else {
                nrm = computeGeometricNormal(mi, primitiveID);
            }
            {
                float3x3 Nmat = float3x3(
                    instanceNormals[0].normalCol0[mi].xyz,
                    instanceNormals[0].normalCol1[mi].xyz,
                    instanceNormals[0].normalCol2[mi].xyz
                );
                nrm = normalize(mul(Nmat, nrm));
            }
            bool entering = dot(nrm, rayD) < -1e-5;
            if (!entering) nrm = -nrm;

            if (sp == 0 && bounce == 0) {
                normalImage[int2(px.x, px.y)] = float4(nrm * 0.5 + 0.5, 1.0);
                depthImage[int2(px.x, px.y)] = tHit;
            }

            uint matIdx = instanceRanges[0].materialID[mi];
            GpuMaterial mat = materials.items[clamp(matIdx, 0u, 7u)];
            int matType = int(mat.params.z);

            if (matType == MATERIAL_CHECKERBOARD) {
                float cx = floor(pos.x * 0.5);
                float cz = floor(pos.z * 0.5);
                bool white = ((int(cx + cz) % 2) == 0);
                float3 checkerAlbedo = white ? float3(0.75, 0.75, 0.75) : float3(0.15, 0.15, 0.15);
                float3 directLin = sampleDirectLighting(pos, nrm, checkerAlbedo, rng);

                BounceState bs;
                bs.rayO = rayO; bs.rayD = rayD;
                bs.throughput = throughput; bs.decayC1 = decayC1; bs.decayC2 = decayC2;
                bs.lastSplit = lastSplit; bs.disp = disp;
                bs.lastGeom = lastGeom; bs.geomD = geomD;
                bs.lamStart = lamStart; bs.lamEnd = lamEnd;
                bs.generation = generation; bs.colorAccum = colorAccum;
                handleCheckerboard(bs, rng, stackTop, rayStack,
                                   pos, nrm, checkerAlbedo, directLin,
                                   bounce, int(params.scatterSamples));
                rayO = bs.rayO; rayD = bs.rayD;
                throughput = bs.throughput; decayC1 = bs.decayC1; decayC2 = bs.decayC2;
                lastSplit = bs.lastSplit; disp = bs.disp;
                lastGeom = bs.lastGeom; geomD = bs.geomD;
                lamStart = bs.lamStart; lamEnd = bs.lamEnd;
                generation = bs.generation; colorAccum = bs.colorAccum;
                if (throughput < 0.005) break;
            }
            else if (matType == MATERIAL_LAMBERTIAN) {
                float3 directLin = sampleDirectLighting(pos, nrm, mat.albedo.xyz, rng);

                BounceState bs;
                bs.rayO = rayO; bs.rayD = rayD;
                bs.throughput = throughput; bs.decayC1 = decayC1; bs.decayC2 = decayC2;
                bs.lastSplit = lastSplit; bs.disp = disp;
                bs.lastGeom = lastGeom; bs.geomD = geomD;
                bs.lamStart = lamStart; bs.lamEnd = lamEnd;
                bs.generation = generation; bs.colorAccum = colorAccum;
                handleLambertian(bs, rng, stackTop, rayStack,
                                 pos, nrm, mat, directLin,
                                 bounce, int(params.scatterSamples));
                rayO = bs.rayO; rayD = bs.rayD;
                throughput = bs.throughput; decayC1 = bs.decayC1; decayC2 = bs.decayC2;
                lastSplit = bs.lastSplit; disp = bs.disp;
                lastGeom = bs.lastGeom; geomD = bs.geomD;
                lamStart = bs.lamStart; lamEnd = bs.lamEnd;
                generation = bs.generation; colorAccum = bs.colorAccum;
                if (throughput < 0.005) break;
            }
            else if (matType == MATERIAL_DIELECTRIC) {
                BounceState bs;
                bs.rayO = rayO; bs.rayD = rayD;
                bs.throughput = throughput; bs.decayC1 = decayC1; bs.decayC2 = decayC2;
                bs.lastSplit = lastSplit; bs.disp = disp;
                bs.lastGeom = lastGeom; bs.geomD = geomD;
                bs.lamStart = lamStart; bs.lamEnd = lamEnd;
                bs.generation = generation; bs.colorAccum = colorAccum;
                handleDielectric(bs, stackTop, rayStack,
                                 pos, nrm, mat, tHit, entering, bounce, sz,
                                 params.fovTan, params.splitMult, params.forceSplitWidth);
                rayO = bs.rayO; rayD = bs.rayD;
                throughput = bs.throughput; decayC1 = bs.decayC1; decayC2 = bs.decayC2;
                lastSplit = bs.lastSplit; disp = bs.disp;
                lastGeom = bs.lastGeom; geomD = bs.geomD;
                lamStart = bs.lamStart; lamEnd = bs.lamEnd;
                generation = bs.generation; colorAccum = bs.colorAccum;
            }
            else if (matType == MATERIAL_METAL) {
                BounceState bs;
                bs.rayO = rayO; bs.rayD = rayD;
                bs.throughput = throughput; bs.decayC1 = decayC1; bs.decayC2 = decayC2;
                bs.lastSplit = lastSplit; bs.disp = disp;
                bs.lastGeom = lastGeom; bs.geomD = geomD;
                bs.lamStart = lamStart; bs.lamEnd = lamEnd;
                bs.generation = generation; bs.colorAccum = colorAccum;
                handleMetal(bs, pos, nrm, mat);
                rayO = bs.rayO; rayD = bs.rayD;
                throughput = bs.throughput; decayC1 = bs.decayC1; decayC2 = bs.decayC2;
                lastSplit = bs.lastSplit; disp = bs.disp;
                lastGeom = bs.lastGeom; geomD = bs.geomD;
                lamStart = bs.lamStart; generation = bs.generation;
                colorAccum = bs.colorAccum;
            }
            else {
                break;
            }
        } // bounce loop
    } // stack loop

        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
            pixelColor += (spectral[b] / 40.0) * BUCKET_CMF[b];
        }
        pixelColor += colorAccum;
    } // SPP loop

    uint outIdx = uint(px.y * sz.x + px.x);
    uint base = outIdx * 4u;
    float3 oldAccum = float3(accumData[base], accumData[base+1], accumData[base+2]);
    float oldCount = accumData[base+3];
    float3 newAccum = oldAccum + pixelColor;
    float newCount = oldCount + float(spp);
    accumData[base]   = newAccum.x;
    accumData[base+1] = newAccum.y;
    accumData[base+2] = newAccum.z;
    accumData[base+3] = newCount;
    float3 outputColor = linearToSRGB(newAccum / max(newCount, 1.0) * 2.5);
    outputImage[int2(px.x, px.y)] = float4(outputColor, 1.0);
}
