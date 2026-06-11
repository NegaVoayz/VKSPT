import common.types;
import common.cmf_data;
import common.bindings;
import math.color;
import math.sampling;
import geometry.normals;
import materials.metal;
import materials.checkerboard;
import materials.lambertian;
import materials.sky_handler;
import materials.dielectric;
import materials.direct_lighting;

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
};
[[vk::push_constant]] PushConstants params;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID) {
    int2 px = int2(dispatchID.xy);
    int w, h; outputImage.GetDimensions(w, h);
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
        RayState init = {
            origin, camDir, float3(0), float3(0), float3(0), float3(0),
            LAMBDA_MIN, LAMBDA_MAX, 1.0, 0.0, 0.0, 0, 0
        };
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
                if (sp == 0 && bounce == 0) {
                    normalImage[int2(px.x, px.y)] = float4(0.0, 0.0, 0.0, 1.0);
                    depthImage[int2(px.x, px.y)] = 1e6;
                }
                BounceState bs = packBs(rayO, rayD, throughput, decayC1, decayC2,
                                        lastSplit, disp, lastGeom, geomD,
                                        lamStart, lamEnd, generation, colorAccum);
                handleSkyMiss(bs, stackTop, rayStack, spectral,
                              rayO, rayD, bounce, sp, px, sz,
                              params.fovTan, params.splitMult);
                unpackBs(rayO, rayD, throughput, decayC1, decayC2,
                         lastSplit, disp, lastGeom, geomD,
                         lamStart, lamEnd, generation, colorAccum, bs);
                break;
            }

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
                nrm = normalize(mul(nrm, Nmat));
            }
            bool entering = dot(nrm, rayD) < 0.0;
            float3 rawNrm = nrm;
            float3 facingNrm = entering ? nrm : -nrm;

            if (sp == 0 && bounce == 0) {
                normalImage[int2(px.x, px.y)] = float4(facingNrm * 0.5 + 0.5, 1.0);
                depthImage[int2(px.x, px.y)] = tHit;
            }

            uint matIdx = instanceRanges[0].materialID[mi];
            GpuMaterial mat = materials.items[clamp(matIdx, 0u, max(0u, uint(params.materialCount) - 1u))];
            int matType = int(mat.params.z);

            if (matType == MATERIAL_CHECKERBOARD) {
                float cx = floor(pos.x * 0.5); float cz = floor(pos.z * 0.5);
                bool white = ((int(cx + cz) % 2) == 0);
                float3 cb = white ? float3(0.75) : float3(0.15);
                float3 dl = sampleDirectLighting(pos, facingNrm, cb, rng, params.diffuseStrength, params.numLights);
                BounceState bs = packBs(rayO, rayD, throughput, decayC1, decayC2,
                                        lastSplit, disp, lastGeom, geomD,
                                        lamStart, lamEnd, generation, colorAccum);
                handleCheckerboard(bs, rng, stackTop, rayStack,
                                   pos, facingNrm, cb, dl, bounce, int(params.scatterSamples));
                unpackBs(rayO, rayD, throughput, decayC1, decayC2,
                         lastSplit, disp, lastGeom, geomD,
                         lamStart, lamEnd, generation, colorAccum, bs);
                if (throughput < 0.005) break;
            }
            else if (matType == MATERIAL_LAMBERTIAN) {
                float3 dl = sampleDirectLighting(pos, facingNrm, mat.albedo.xyz, rng, params.diffuseStrength, params.numLights);
                BounceState bs = packBs(rayO, rayD, throughput, decayC1, decayC2,
                                        lastSplit, disp, lastGeom, geomD,
                                        lamStart, lamEnd, generation, colorAccum);
                handleLambertian(bs, rng, stackTop, rayStack,
                                 pos, facingNrm, mat, dl, bounce, int(params.scatterSamples));
                unpackBs(rayO, rayD, throughput, decayC1, decayC2,
                         lastSplit, disp, lastGeom, geomD,
                         lamStart, lamEnd, generation, colorAccum, bs);
                if (throughput < 0.005) break;
            }
            else if (matType == MATERIAL_DIELECTRIC) {
                BounceState bs = packBs(rayO, rayD, throughput, decayC1, decayC2,
                                        lastSplit, disp, lastGeom, geomD,
                                        lamStart, lamEnd, generation, colorAccum);
                handleDielectric(bs, stackTop, rayStack,
                                 pos, rawNrm, mat, tHit, bounce, sz,
                                 params.fovTan, params.splitMult, params.forceSplitWidth);
                unpackBs(rayO, rayD, throughput, decayC1, decayC2,
                         lastSplit, disp, lastGeom, geomD,
                         lamStart, lamEnd, generation, colorAccum, bs);
            }
            else if (matType == MATERIAL_METAL) {
                BounceState bs = packBs(rayO, rayD, throughput, decayC1, decayC2,
                                        lastSplit, disp, lastGeom, geomD,
                                        lamStart, lamEnd, generation, colorAccum);
                handleMetal(bs, pos, facingNrm, mat);
                unpackBs(rayO, rayD, throughput, decayC1, decayC2,
                         lastSplit, disp, lastGeom, geomD,
                         lamStart, lamEnd, generation, colorAccum, bs);
            }
            else { break; }
        } // bounce
    } // stack

        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            pixelColor += (spectral[b] / 40.0) * BUCKET_CMF[b];
        pixelColor += colorAccum;
    } // spp

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
