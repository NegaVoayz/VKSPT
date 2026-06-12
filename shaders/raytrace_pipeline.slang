// Full spectral RT pipeline — Legendre log-decay + dispersion + sky splitting
import common.types;
import common.cmf_data;
import common.bindings;
import geometry.normals;
import geometry.dispersion;
import math.sampling;
import math.spectral_math;
import math.legendre_decay;
import math.color;

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
    float minSplitNm;    // stop split when λ range < this (nm)
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

// ===== DIRECT LIGHTING =====
float3 directLight(float3 pos, float3 nrm, float3 albedo, inout uint rng)
{
    float3 contrib = float3(0.0);
    for (int li = 0; li < params.numLights; li++) {
        GpuLight l = lights.items[li];
        int lt = int(l.pos_type.w);
        float3 toLight; float tM, att, spA = 1.0;
        if (lt == LIGHT_DIRECTIONAL) {
            toLight = l.pos_type.xyz; tM = 10000.0;
            att = l.color_intensity.w * 0.01;
        } else {
            toLight = l.pos_type.xyz - pos; float d = length(toLight);
            toLight /= d; tM = d - 0.001;
            att = l.color_intensity.w / max(d*d, 1.0);
            if (lt == LIGHT_SPOT) {
                float cA = dot(-toLight, l.dir_inner.xyz);
                spA = smoothstep(l.outer_range.x, l.outer_range.x+0.05, cA);
            }
        }
        if (att < 0.0001 || spA < 0.001) continue;
        float nDL = max(dot(nrm, toLight), 0.0);
        if (nDL <= 0.0) continue;

        Payload sp; sp.isShadowed = 0.0; sp.bounce = -1; sp.rng = rng;
        RayDesc sr; sr.Origin = pos + nrm*0.001; sr.Direction = toLight;
        sr.TMin = 0.001; sr.TMax = tM;
        TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
                 RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 0, 0, sr, sp);
        if (sp.isShadowed > 0.5) {
            float3 lc = l.color_intensity.xyz;
            contrib += lc * (albedo * 0.31831) * (att * nDL * spA) * params.diffuseStrength;
        }
    }
    return contrib;
}

// ===== SKY SPECTRUM ACCUMULATION =====
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

// ===== RAY GENERATION =====
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
        initPayload(p, LAMBDA_MIN, LAMBDA_MAX, 0.0, 0.0, 0.0, 0, 0, seed);

        RayDesc ray;
        ray.Origin = origin; ray.Direction = camDir;
        ray.TMin = 0.001; ray.TMax = 10000.0;
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

// ===== CLOSEST HIT =====
[shader("closesthit")]
void ClosestHitMain(inout Payload p, in BuiltInTriangleIntersectionAttributes attr)
{
    p.hitT = RayTCurrent();
    float3 pos = WorldRayOrigin() + WorldRayDirection() * p.hitT;
    int mi = InstanceIndex(), pid = PrimitiveIndex();

    float3 nrm;
    if (instanceRanges[0].useSmoothNormals[mi] != 0u)
        nrm = computeSmoothNormal(mi, pid, attr.barycentrics);
    else nrm = computeGeometricNormal(mi, pid);
    float3x3 Nm = float3x3(instanceNormals[0].normalCol0[mi].xyz,
                         instanceNormals[0].normalCol1[mi].xyz,
                         instanceNormals[0].normalCol2[mi].xyz);
    nrm = normalize(mul(nrm, Nm));
    float3 fNrm = dot(nrm, WorldRayDirection()) < 0.0 ? nrm : -nrm;
    p.normal = fNrm;

    uint matIdx = instanceRanges[0].materialID[mi];
    GpuMaterial mat = materials.items[clamp(matIdx, 0u,
        max(0u, uint(params.materialCount)-1u))];
    int matType = int(mat.params.z);

    // Accumulate dispersion over travel distance
    p.lastSplit += p.disp * p.hitT;

    // RR
    if (p.bounce >= 3) {
        float linTp = exp(p.throughput - 0.5*p.decayC2);
        float survive = max(linTp, 0.1);
        if (rand1(p.rng) > survive) { p.color = float3(0.0); return; }
        p.throughput -= log(survive);
    }
    if (p.bounce >= params.maxBounces || decayIsDead(p.throughput, p.decayC2))
        { p.color = float3(0.0); return; }

    if (matType == MATERIAL_CHECKERBOARD) {
        float cx = floor(pos.x*0.5), cz = floor(pos.z*0.5);
        float3 alb = ((int(cx+cz)%2)==0) ? float3(0.75) : float3(0.15);
        p.color = exp(p.throughput) * directLight(pos, fNrm, alb, p.rng);

        float3 sDir = cosineSampleHemisphere(fNrm, p.rng);
        Payload np; initPayload(np, p.lamStart, p.lamEnd,
            p.throughput + log(dot(alb, float3(0.333))),
            p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
        RayDesc nr; nr.Origin=pos+fNrm*0.001+sDir*0.01; nr.Direction=sDir;
        nr.TMin=0.001; nr.TMax=10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
        mergeChild(p, np, alb);
    }
    else if (matType == MATERIAL_LAMBERTIAN) {
        float3 alb = mat.albedo.xyz;
        p.color = exp(p.throughput) * directLight(pos, fNrm, alb, p.rng);

        float3 sDir = cosineSampleHemisphere(fNrm, p.rng);
        Payload np; initPayload(np, p.lamStart, p.lamEnd,
            p.throughput + log(dot(alb, float3(0.333))),
            p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
        RayDesc nr; nr.Origin=pos+fNrm*0.001+sDir*0.01; nr.Direction=sDir;
        nr.TMin=0.001; nr.TMax=10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
        mergeChild(p, np, alb);
    }
    else if (matType == MATERIAL_METAL) {
        float3 ma = mat.albedo.xyz;
        float r0 = ma.y + (ma.x+ma.z-2.0*ma.y)/6.0;
        float r1 = (ma.x - ma.z)*0.5;
        float r2 = (ma.x+ma.z-2.0*ma.y)/3.0;
        float lamMid = float(p.lamStart+p.lamEnd)*0.5;
        float halfW  = float(p.lamEnd-p.lamStart)*0.5;
        float LAM_C = float(LAMBDA_MIN+LAMBDA_MAX)*0.5;
        float LAM_H = float(LAMBDA_MAX-LAMBDA_MIN)*0.5;
        const float SQ3_5 = 0.77459667;
        const float W0=5.0/9.0, W1=8.0/9.0, W2=5.0/9.0;

        float newL[3];
        for (int k=0; k<3; k++) {
            float xk = (k==0)?-SQ3_5:((k==1)?0.0:SQ3_5);
            float Lk = p.throughput + p.decayC1*xk + p.decayC2*(1.5*xk*xk-0.5);
            float lam_k = lamMid + halfW*xk;
            float Rk = evalLegendre(r0, r1, r2, (lam_k-LAM_C)/LAM_H);
            newL[k] = Lk + log(max(Rk, 1e-10));
        }
        float newC0=0.5*(W0*newL[0]+W1*newL[1]+W2*newL[2]);
        float newC1=1.5*(W0*newL[0]*(-SQ3_5)+W2*newL[2]*SQ3_5);
        float p2_0=0.4, p2_1=-0.5, p2_2=0.4;
        float newC2=2.5*(W0*newL[0]*p2_0+W1*newL[1]*p2_1+W2*newL[2]*p2_2);

        float3 refl = reflectDir(WorldRayDirection(), fNrm);
        Payload np; initPayload(np, p.lamStart, p.lamEnd, newC0, newC1, newC2,
                                p.bounce+1, p.generation, p.rng);
        np.lastSplit = p.lastSplit - 2.0*dot(p.lastSplit, fNrm)*fNrm;
        np.disp      = p.disp      - 2.0*dot(p.disp, fNrm)*fNrm;
        RayDesc nr; nr.Origin=pos+fNrm*0.001+refl*0.01; nr.Direction=refl;
        nr.TMin=0.001; nr.TMax=10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
        mergeChild(p, np, float3(1.0));
    }
    else if (matType == MATERIAL_DIELECTRIC) {
        bool entering = dot(nrm, WorldRayDirection()) < 0.0;
        float3 dn = entering ? nrm : -nrm;

        int lamMid = (p.lamStart+p.lamEnd)/2;
        float ior_start = cauchyIOR(mat, float(p.lamStart));
        float ior_end   = cauchyIOR(mat, float(p.lamEnd));
        float ior_mid   = cauchyIOR(mat, float(lamMid));
        float eta_mid   = entering ? (1.0/ior_mid) : ior_mid;
        float deta      = entering ? (1.0/ior_end-1.0/ior_start) : (ior_end-ior_start);

        float3 newDir;
        if (refractDir(WorldRayDirection(), dn, eta_mid, newDir)) {
            float3 newDisp = computeDispersion(WorldRayDirection(), dn, eta_mid, deta);

            float3 rotSplit = rotateToNewFrame(p.lastSplit, WorldRayDirection(), newDir);
            float3 rotDisp  = rotateToNewFrame(p.disp, WorldRayDirection(), newDir);

            float cosTheta = abs(dot(WorldRayDirection(), dn));
            float R0 = (ior_mid-1.0)/(ior_mid+1.0); R0 *= R0;
            float fresnelR = R0 + (1.0-R0)*pow(1.0-cosTheta, 5.0);

            if (fresnelR > 0.001 && !decayIsDead(p.throughput+log(fresnelR), p.decayC2)) {
                float3 refl = reflectDir(WorldRayDirection(), dn);
                Payload rp; initPayload(rp, p.lamStart, p.lamEnd,
                    p.throughput+log(max(fresnelR,1e-10)),
                    p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
                rp.lastSplit = p.lastSplit-2.0*dot(p.lastSplit,dn)*dn;
                rp.disp      = p.disp-2.0*dot(p.disp,dn)*dn;
                RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
                rr.TMin=0.001; rr.TMax=10000.0;
                TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
                mergeChild(p, rp, float3(1.0));
            }

            float refrTp = p.throughput + log(max(1.0-fresnelR, 1e-10));
            if (!decayIsDead(refrTp, p.decayC2)) {
                // Check if dielectric split is needed at this interface
                int lamRange = p.lamEnd - p.lamStart;
                float estDist = max(p.hitT, 50.0);
                float3 predictedSep = rotSplit + (rotDisp + newDisp) * estDist;
                float sepMag = length(predictedSep);
                int minRange = max(int(params.minSplitNm), 20);
                if (sepMag > params.forceSplitWidth && params.forceSplitWidth > 0.0 &&
                    p.generation < 7) {

                    int midWl = (p.lamStart + p.lamEnd)/2;
                    float3 perp = normalize(predictedSep);
                    if (length(perp) < 0.0001) perp = float3(1,0,0);

                    // Left (shorter λ)
                    {
                        float3 vdir = normalize(newDir + perp * sepMag * 0.5 / estDist);
                        Payload vp; initPayload(vp, p.lamStart, midWl,
                            0,0,0, p.bounce+1, p.generation+1, p.rng);
                        reprojectDecayLeft(refrTp, p.decayC1, p.decayC2,
                                           vp.throughput, vp.decayC1, vp.decayC2);
                        vp.lastSplit = rotSplit; vp.disp = rotDisp + newDisp;
                        RayDesc vr; vr.Origin=pos-dn*0.001+vdir*0.01; vr.Direction=vdir;
                        vr.TMin=0.001; vr.TMax=10000.0;
                        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, vr, vp);
                        mergeChild(p, vp, float3(1.0));
                    }
                    // Right (longer λ)
                    {
                        float3 rdir = normalize(newDir - perp * sepMag * 0.5 / estDist);
                        Payload rp; initPayload(rp, midWl, p.lamEnd,
                            0,0,0, p.bounce+1, p.generation+1, p.rng);
                        reprojectDecayRight(refrTp, p.decayC1, p.decayC2,
                                            rp.throughput, rp.decayC1, rp.decayC2);
                        rp.lastSplit = rotSplit; rp.disp = rotDisp + newDisp;
                        RayDesc rr; rr.Origin=pos-dn*0.001+rdir*0.01; rr.Direction=rdir;
                        rr.TMin=0.001; rr.TMax=10000.0;
                        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
                        mergeChild(p, rp, float3(1.0));
                    }
                } else {
                        Payload tp; initPayload(tp, p.lamStart, p.lamEnd,
                            refrTp, p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
                        tp.lastSplit = rotSplit;
                        tp.disp      = rotDisp + newDisp;
                        RayDesc tr; tr.Origin=pos-dn*0.001+newDir*0.01; tr.Direction=newDir;
                        tr.TMin=0.001; tr.TMax=10000.0;
                        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, tr, tp);
                        mergeChild(p, tp, float3(1.0));
                }
            }
        } else {
            // TIR: 100% reflection
            float3 refl = reflectDir(WorldRayDirection(), dn);
            Payload rp; initPayload(rp, p.lamStart, p.lamEnd,
                p.throughput, p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
            rp.lastSplit = p.lastSplit-2.0*dot(p.lastSplit,dn)*dn;
            rp.disp      = p.disp-2.0*dot(p.disp,dn)*dn;
            RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
            rr.TMin=0.001; rr.TMax=10000.0;
            TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
            mergeChild(p, rp, float3(1.0));
        }
    }
    else { p.color = float3(0.5); }
}

// ===== MISS (sky split via recursive TraceRay) =====
[shader("miss")]
void MissMain(inout Payload p)
{
    if (p.bounce < 0) { p.isShadowed = 1.0; return; }

    float3 dir = WorldRayDirection();
    float estSkyDist = 10000.0;
    float3 skyDisp   = p.lastSplit + p.disp * estSkyDist;
    float skySepMag  = length(skyDisp);

    int lamRange = p.lamEnd - p.lamStart;
    int minRange = max(int(params.minSplitNm), 20);
    if (skySepMag > params.forceSplitWidth && params.forceSplitWidth > 0.0 &&
        lamRange >= minRange &&
        !decayIsDead(p.throughput, p.decayC2) && p.generation < 7) {

        int midWl = (p.lamStart + p.lamEnd)/2;
        float3 skyPerp = normalize(skyDisp);
        if (length(skyPerp) < 0.0001) skyPerp = float3(1,0,0);

        // Left (shorter λ)
        {
            float3 vdir = normalize(dir + skyPerp * skySepMag * 0.5 / estSkyDist);
            Payload vp; initPayload(vp, p.lamStart, midWl,
                0,0,0, p.bounce, p.generation+1, p.rng);
            reprojectDecayLeft(p.throughput, p.decayC1, p.decayC2,
                               vp.throughput, vp.decayC1, vp.decayC2);
            vp.lastSplit = p.lastSplit; vp.disp = p.disp;
            RayDesc vr; vr.Origin = WorldRayOrigin(); vr.Direction = vdir;
            vr.TMin = 0.001; vr.TMax = 10000.0;
            TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, vr, vp);
            mergeChild(p, vp, float3(1.0));
        }
        // Right (longer λ)
        {
            float3 rdir = normalize(dir - skyPerp * skySepMag * 0.5 / estSkyDist);
            Payload rp; initPayload(rp, midWl, p.lamEnd,
                0,0,0, p.bounce, p.generation+1, p.rng);
            reprojectDecayRight(p.throughput, p.decayC1, p.decayC2,
                                rp.throughput, rp.decayC1, rp.decayC2);
            rp.lastSplit = p.lastSplit; rp.disp = p.disp;
            RayDesc rr; rr.Origin = WorldRayOrigin(); rr.Direction = rdir;
            rr.TMin = 0.001; rr.TMax = 10000.0;
            TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
            mergeChild(p, rp, float3(1.0));
        }
        p.hitT   = 1e6;
        p.normal = float3(0,0,1);
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
    p.hitT   = 1e6;
    p.normal = float3(0,0,1);
}
