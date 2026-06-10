import common.types;
import common.cmf_data;
import math.spectral_math;
import math.color;
import math.absorption;
import math.legendre_decay;
import math.sampling;
import geometry.dispersion;
import geometry.normals;

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

float sampleLightSPD(int refIdx) {
    return lights.items[0].color_intensity.x;
}

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

void accumulateSpectrum(inout float spectral[NUM_SPECTRAL_BUCKETS],
                         float lamMid, float halfWidth,
                         float c0, float c1, float c2,
                         float3 envLinear) {
    float lamS = lamMid - halfWidth;
    float lamE = lamMid + halfWidth;
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

                float estSkyDist = 10000.0;
                float3  skyDisp    = lastSplit + disp * estSkyDist;
                float skySepMag  = length(skyDisp);
                float skyPixHW   = params.fovTan / float(sz.x);
                float skyThresh  = estSkyDist * skyPixHW * 0.5 * params.splitMult;

                int lamRange = lamEnd - lamStart;
                if (skySepMag > skyThresh && lamRange >= 20 && throughput >= 0.05 && generation < 7) {
                    int midWl = (lamStart + lamEnd) / 2;
                    float3 skyPerp = normalize(skyDisp);
                    if (length(skyPerp) < 0.0001) skyPerp = float3(1,0,0);

                    float splitParentMid  = float(lamStart + lamEnd) * 0.5;
                    float splitParentHalf = float(lamEnd - lamStart) * 0.5;
                    float violetMid  = float(lamStart + midWl) * 0.5;
                    float violetHalf = float(midWl - lamStart) * 0.5;

                    if (stackTop < SPECTRAL_STACK_MAX - 1) {
                        RayState skyV;
                        skyV.direction   = normalize(rayD + skyPerp * skySepMag * 0.5 / estSkyDist);
                        skyV.origin      = rayO;
                        skyV.last_split  = lastSplit;
                        skyV.dispersion  = disp;
                        skyV.lam_start   = lamStart;
                        skyV.lam_end     = midWl;
                        reprojectDecay(throughput, decayC1, decayC2,
                                       splitParentMid, splitParentHalf,
                                       violetMid, violetHalf,
                                       skyV.throughput, skyV.decay_c1, skyV.decay_c2);
                        skyV.bounce      = bounce;
                        skyV.generation  = generation + 1;
                        rayStack[stackTop++] = skyV;

                        RayState skyR;
                        skyR.direction   = normalize(rayD - skyPerp * skySepMag * 0.5 / estSkyDist);
                        skyR.origin      = rayO;
                        skyR.last_split  = lastSplit;
                        skyR.dispersion  = disp;
                        skyR.lam_start   = midWl;
                        skyR.lam_end     = lamEnd;
                        float redMid  = float(midWl + lamEnd) * 0.5;
                        float redHalf = float(lamEnd - midWl) * 0.5;
                        reprojectDecay(throughput, decayC1, decayC2,
                                       splitParentMid, splitParentHalf,
                                       redMid, redHalf,
                                       skyR.throughput, skyR.decay_c1, skyR.decay_c2);
                        skyR.bounce      = bounce;
                        skyR.generation  = generation + 1;
                        rayStack[stackTop++] = skyR;
                        break;
                    }
                }

                float lamMid  = float(lamStart + lamEnd) * 0.5;
                float halfW   = float(lamEnd - lamStart) * 0.5;
                float3 envLinear = srgbToLinear(float3(0.5));
                accumulateSpectrum(spectral, lamMid, halfW, throughput, decayC1, decayC2, envLinear);
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
                float preAbsTP = throughput;
                throughput *= (0.07 + 0.93 * dot(checkerAlbedo, float3(0.299, 0.587, 0.114)));
                float3 directLin = sampleDirectLighting(pos, nrm, checkerAlbedo, rng);
                colorAccum += directLin * preAbsTP;
                if (bounce >= 3) {
                    float survive = max(throughput, 0.1);
                    if (rand1(rng) > survive) break;
                    throughput /= survive;
                }
                int scatCount = int(params.scatterSamples);
                if (scatCount < 1) scatCount = 1;
                if (scatCount > 4) scatCount = 4;
                float scatThroughput = throughput / float(scatCount);
                float scatC1 = decayC1 / float(scatCount);
                float scatC2 = decayC2 / float(scatCount);
                for (int s = 0; s < scatCount - 1; s++) {
                    if (stackTop < SPECTRAL_STACK_MAX) {
                        float3 sDir = cosineSampleHemisphere(nrm, rng);
                        RayState scat;
                        scat.origin      = pos + nrm * 0.001 + sDir * 0.01;
                        scat.direction   = sDir;
                        scat.last_split  = float3(0.0); scat.dispersion  = float3(0.0);
                        scat.last_geom   = float3(0.0); scat.geomDisp    = float3(0.0);
                        scat.lam_start   = lamStart;  scat.lam_end     = lamEnd;
                        scat.throughput      = scatThroughput;
                        scat.decay_c1    = scatC1;     scat.decay_c2    = scatC2;
                        scat.bounce      = bounce + 1; scat.generation  = generation;
                        rayStack[stackTop++] = scat;
                    }
                }
                rayD       = cosineSampleHemisphere(nrm, rng);
                rayO       = pos + nrm * 0.001 + rayD * 0.01;
                throughput     = scatThroughput; decayC1 = scatC1; decayC2 = scatC2;
                lastSplit  = float3(0.0); disp = float3(0.0);
                lastGeom   = float3(0.0); geomD = float3(0.0);
            }
            else if (matType == MATERIAL_LAMBERTIAN) {
                float nl = rgbToDomLambda(mat.albedo.xyz);
                float lum = dot(mat.albedo.xyz, float3(0.299, 0.587, 0.114));
                float maxC = max(max(mat.albedo.xyz.x, mat.albedo.xyz.y), mat.albedo.xyz.z);
                float minC = min(min(mat.albedo.xyz.x, mat.albedo.xyz.y), mat.albedo.xyz.z);
                float sat = (maxC - minC) / max(maxC, 0.001);

                float lamS = float(lamStart), lamE = float(lamEnd);
                float lamM = (lamS + lamE) * 0.5;
                float dS = lamS - nl, dM = lamM - nl, dE = lamE - nl;
                float attAvg = (exp(-0.5*dS*dS/7200.0) + exp(-0.5*dM*dM/7200.0) + exp(-0.5*dE*dE/7200.0)) / 3.0;

                float preAbsTP = throughput;
                float attMix = lerp(1.0, attAvg, sat);
                throughput *= attMix * (0.07 + 0.93 * lum);

                float3 directLin = sampleDirectLighting(pos, nrm, mat.albedo.xyz, rng);
                colorAccum += directLin * preAbsTP;

                if (bounce >= 3) {
                    float survive = max(throughput, 0.1);
                    if (rand1(rng) > survive) break;
                    throughput /= survive;
                }

                int scatCount = int(params.scatterSamples);
                if (scatCount < 1) scatCount = 1;
                if (scatCount > 4) scatCount = 4;
                float scatThroughput = throughput / float(scatCount);
                float scatC1 = decayC1 / float(scatCount);
                float scatC2 = decayC2 / float(scatCount);
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
                        scat.lam_start   = lamStart;
                        scat.lam_end     = lamEnd;
                        scat.throughput      = scatThroughput;
                        scat.decay_c1    = scatC1;
                        scat.decay_c2    = scatC2;
                        scat.bounce      = bounce + 1;
                        scat.generation  = generation;
                        rayStack[stackTop++] = scat;
                    }
                }
                rayD       = cosineSampleHemisphere(nrm, rng);
                rayO       = pos + nrm * 0.001 + rayD * 0.01;
                throughput     = scatThroughput;
                decayC1    = scatC1;
                decayC2    = scatC2;
                lastSplit  = float3(0.0); disp = float3(0.0);
                lastGeom   = float3(0.0); geomD = float3(0.0);
            }
            else if (matType == MATERIAL_DIELECTRIC) {
                int lamMid = (lamStart + lamEnd) / 2;
                float ior_start = cauchyIOR(mat, float(lamStart));
                float ior_end   = cauchyIOR(mat, float(lamEnd));
                float ior_mid   = cauchyIOR(mat, float(lamMid));
                float eta_mid   = entering ? (1.0 / ior_mid) : ior_mid;
                float deta      = (entering ? (1.0/ior_end - 1.0/ior_start) : (ior_end - ior_start));

                float lamMidF  = float(lamStart + lamEnd) * 0.5;
                float halfW    = float(lamEnd - lamStart) * 0.5;
                bool decayPoorFit = applyVolumeDecay(throughput, decayC1, decayC2,
                                                      lamMidF, halfW,
                                                      mat.absorpA.y, mat.absorpB.y, tHit);

                float3 newDir;
                if (refractDir(rayD, nrm, eta_mid, newDir)) {
                    float3 newDisp = computeDispersion(rayD, nrm, eta_mid, deta);
                    float3 newGeom = computeGeomDispersion(rayD, nrm, eta_mid);

                    lastSplit = rotateToNewFrame(lastSplit, rayD, newDir);
                    disp      = rotateToNewFrame(disp, rayD, newDir) + newDisp;
                    lastGeom  = rotateToNewFrame(lastGeom, rayD, newDir);
                    geomD     = rotateToNewFrame(geomD, rayD, newDir) + newGeom;

                    float cosTheta = abs(dot(rayD, nrm));
                    float R0 = ((ior_mid - 1.0) / (ior_mid + 1.0));
                    R0 = R0 * R0;
                    float fresnelR = R0 + (1.0 - R0) * pow(1.0 - cosTheta, 5.0);

                    float reflThroughput = throughput * fresnelR;
                    if (stackTop < SPECTRAL_STACK_MAX && reflThroughput > 0.005) {
                        float3 reflDir = reflectDir(rayD, nrm);
                        RayState refl;
                        refl.origin      = pos + nrm * 0.001 + reflDir * 0.01;
                        refl.direction   = reflDir;
                        refl.last_split  = lastSplit - 2.0 * dot(lastSplit, nrm) * nrm;
                        refl.dispersion  = disp - 2.0 * dot(disp, nrm) * nrm;
                        refl.last_geom   = lastGeom - 2.0 * dot(lastGeom, nrm) * nrm;
                        refl.geomDisp    = geomD - 2.0 * dot(geomD, nrm) * nrm;
                        refl.lam_start   = lamStart;
                        refl.lam_end     = lamEnd;
                        refl.throughput      = reflThroughput;
                        refl.decay_c1    = decayC1 * fresnelR;
                        refl.decay_c2    = decayC2 * fresnelR;
                        refl.bounce      = bounce + 1;
                        refl.generation  = generation;
                        rayStack[stackTop++] = refl;
                    }
                    throughput *= (1.0 - fresnelR);
                    decayC1    *= (1.0 - fresnelR);
                    decayC2    *= (1.0 - fresnelR);

                    float estDist = max(tHit, 50.0);
                    float3  predictedGeom = lastGeom + geomD * estDist;
                    float3  predictedSep  = lastSplit + disp * estDist;
                    float sepMag       = max(length(predictedGeom), length(predictedSep));
                    float pixHalfWidth = params.fovTan / float(sz.x);
                    float pixThresh    = estDist * pixHalfWidth * 0.5 * params.splitMult;

                    int lamRange = lamEnd - lamStart;
                    float alphaStart = computeAbsorption(mat, float(lamStart));
                    float alphaEnd   = computeAbsorption(mat, float(lamEnd));
                    float alphaVar   = abs(exp(-alphaStart * tHit) - exp(-alphaEnd * tHit));

                    bool forceWide = (int(params.forceSplitWidth) > 0) && (lamRange > int(params.forceSplitWidth));
                    bool needSplit = (sepMag > pixThresh) || (alphaVar > 0.05 && decayPoorFit) || forceWide;

                    bool spectralDriven = (length(predictedSep) > pixThresh);
                    bool geomDriven     = (length(predictedGeom) > pixThresh);

                    if (needSplit && lamRange >= 10 && throughput >= 0.05 && generation < 8) {
                        int midWl = (lamStart + lamEnd) / 2;
                        int nextGen = generation + 1;

                        if (geomDriven && !spectralDriven) {
                            nextGen = generation + 4;
                            RayState gchild;
                            gchild.origin      = pos + nrm * 0.001 + newDir * 0.01;
                            gchild.direction   = newDir;
                            gchild.last_split  = predictedSep;
                            gchild.dispersion  = disp;
                            gchild.last_geom   = predictedGeom;
                            gchild.geomDisp    = geomD;
                            gchild.lam_start   = lamStart;
                            gchild.lam_end     = lamEnd;
                            gchild.throughput      = throughput * 0.5;
                            gchild.decay_c1    = decayC1 * 0.5;
                            gchild.decay_c2    = decayC2 * 0.5;
                            gchild.bounce      = bounce + 1;
                            gchild.generation  = nextGen;
                            if (stackTop < SPECTRAL_STACK_MAX) {
                                rayStack[stackTop++] = gchild;
                                throughput *= 0.5; decayC1 *= 0.5; decayC2 *= 0.5;
                                generation = nextGen;
                            }
                        } else {
                            float splitParentMid  = float(lamStart + lamEnd) * 0.5;
                            float splitParentHalf = float(lamEnd - lamStart) * 0.5;
                            float violetMid       = float(lamStart + midWl) * 0.5;
                            float violetHalf      = float(midWl - lamStart) * 0.5;

                            RayState violet;
                            violet.origin      = pos + nrm * 0.001 + newDir * 0.01;
                            violet.direction   = newDir;
                            violet.last_split  = predictedSep;
                            violet.dispersion  = disp;
                            violet.last_geom   = predictedGeom;
                            violet.geomDisp    = geomD;
                            violet.lam_start   = lamStart;
                            violet.lam_end     = midWl;
                            reprojectDecay(throughput, decayC1, decayC2,
                                           splitParentMid, splitParentHalf,
                                           violetMid, violetHalf,
                                           violet.throughput, violet.decay_c1, violet.decay_c2);
                            violet.bounce      = bounce + 1;
                            violet.generation  = nextGen;

                            if (stackTop < SPECTRAL_STACK_MAX) {
                                rayStack[stackTop++] = violet;
                                float redMid  = float(midWl + lamEnd) * 0.5;
                                float redHalf = float(lamEnd - midWl) * 0.5;
                                reprojectDecay(throughput, decayC1, decayC2,
                                               splitParentMid, splitParentHalf,
                                               redMid, redHalf,
                                               throughput, decayC1, decayC2);
                                lamStart   = midWl;
                                generation = nextGen;
                            }
                        }
                    }

                    rayD    = newDir;
                    rayO    = pos + nrm * 0.001 + newDir * 0.01;
                }
                else {
                    newDir     = reflectDir(rayD, nrm);
                    lastSplit  = lastSplit - 2.0 * dot(lastSplit, nrm) * nrm;
                    disp       = disp - 2.0 * dot(disp, nrm) * nrm;
                    lastGeom   = lastGeom - 2.0 * dot(lastGeom, nrm) * nrm;
                    geomD      = geomD - 2.0 * dot(geomD, nrm) * nrm;
                    rayD       = newDir;
                    rayO       = pos + nrm * 0.001 + newDir * 0.01;
                }
            }
            else if (matType == MATERIAL_METAL) {
                float3 ma = mat.albedo.xyz;
                float r0 = ma.y + (ma.x + ma.z - 2.0*ma.y) / 6.0;
                float r1 = (ma.x - ma.z) * 0.5;
                float r2 = (ma.x + ma.z - 2.0*ma.y) / 3.0;
                float lamMidF = float(lamStart + lamEnd) * 0.5;
                float halfWF  = float(lamEnd - lamStart) * 0.5;
                const float SQRT_3_5 = 0.77459667;
                const float W0 = 5.0/9.0, W1 = 8.0/9.0, W2 = 5.0/9.0;
                float newD[3];
                for (int k = 0; k < 3; k++) {
                    float xk = (k == 0) ? -SQRT_3_5 : ((k == 1) ? 0.0 : SQRT_3_5);
                    float Dk = evalDecay(throughput, decayC1, decayC2, xk);
                    float Rk = evalDecay(r0, r1, r2, xk);
                    newD[k] = Dk * Rk;
                }
                throughput = 0.5 * (W0*newD[0] + W1*newD[1] + W2*newD[2]);
                decayC1    = 1.5 * (W0*newD[0]*(-SQRT_3_5) + W2*newD[2]*SQRT_3_5);
                float p2_0 = 0.4, p2_1 = -0.5, p2_2 = 0.4;
                decayC2    = 2.5 * (W0*newD[0]*p2_0 + W1*newD[1]*p2_1 + W2*newD[2]*p2_2);
                float3 reflDir = reflectDir(rayD, nrm);
                rayD       = reflDir;
                rayO       = pos + nrm * 0.001 + reflDir * 0.01;
                lastSplit  = lastSplit - 2.0 * dot(lastSplit, nrm) * nrm;
                disp       = disp - 2.0 * dot(disp, nrm) * nrm;
                lastGeom   = lastGeom - 2.0 * dot(lastGeom, nrm) * nrm;
                geomD      = geomD - 2.0 * dot(geomD, nrm) * nrm;
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
