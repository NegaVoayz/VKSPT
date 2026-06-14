// RT pipeline material handlers — each uses Payload + TraceRay recursion
import common.types;
import common.cmf_data;
import common.bindings;
import geometry.normals;
import geometry.dispersion;
import math.sampling;
import math.spectral_math;
import math.color;
import common.hash_utils;
import rt.payload;
import rt.direct_light;

// ---- Checkerboard ----
void handleCheckerboard(inout Payload p, float3 pos, float3 fNrm, inout uint rng)
{
    float cx = floor(pos.x*0.5), cz = floor(pos.z*0.5);
    float3 alb = ((int(cx+cz)%2)==0) ? float3(0.75) : float3(0.15);

    gatherPhotons(pos, fNrm, p.spectral);

    float3 sDir = cosineSampleHemisphere(fNrm, rng);
    Payload np; initPayload(np, p.lamStart, p.lamEnd,
        p.throughput + log(dot(alb, float3(0.333))),
        p.decayC1, p.decayC2, p.bounce+1, p.generation,
        p.specularChain, p.lightRGB, rng);
    incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
    RayDesc nr; nr.Origin=pos+fNrm*0.001+sDir*0.01; nr.Direction=sDir;
    nr.TMin=0.001; nr.TMax=10000.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
    mergeChild(p, np, alb);
}

// ---- Lambertian ----
void handleLambertian(inout Payload p, float3 pos, float3 fNrm,
                       GpuMaterial mat, inout uint rng)
{
    float3 alb = mat.albedo.xyz;

    // Illumination comes from photon gather (fills p.spectral[])
    gatherPhotons(pos, fNrm, p.spectral);

    float3 sDir = cosineSampleHemisphere(fNrm, rng);
    Payload np; initPayload(np, p.lamStart, p.lamEnd,
        p.throughput + log(dot(alb, float3(0.333))),
        p.decayC1, p.decayC2, p.bounce+1, p.generation,
        p.specularChain, p.lightRGB, rng);
    incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
    RayDesc nr; nr.Origin=pos+fNrm*0.001+sDir*0.01; nr.Direction=sDir;
    nr.TMin=0.001; nr.TMax=10000.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
    mergeChild(p, np, alb);
}

// ---- Metal ----
void handleMetal(inout Payload p, float3 pos, float3 fNrm,
                  GpuMaterial mat, inout uint rng)
{
    float3 ma = mat.albedo.xyz;
    float r0 = ma.y + (ma.x+ma.z-2.0*ma.y)/6.0;
    float r1 = (ma.x - ma.z)*0.5;
    float r2 = (ma.x+ma.z-2.0*ma.y)/3.0;
    float lamMid = float(p.lamStart+p.lamEnd)*0.5;
    float halfW  = float(p.lamEnd-p.lamStart)*0.5;
    float lamC = float(LAMBDA_MIN+LAMBDA_MAX)*0.5;
    float lamH = float(LAMBDA_MAX-LAMBDA_MIN)*0.5;
    const float SQ3_5 = 0.77459667;
    const float W0=5.0/9.0, W1=8.0/9.0, W2=5.0/9.0;

    float newL[3];
    for (int k=0; k<3; k++) {
        float xk = (k==0)?-SQ3_5:((k==1)?0.0:SQ3_5);
        float Lk = p.throughput + p.decayC1*xk + p.decayC2*(1.5*xk*xk-0.5);
        float lam_k = lamMid + halfW*xk;
        float Rk = evalLegendre(r0, r1, r2, (lam_k-lamC)/lamH);
        newL[k] = Lk + log(max(Rk, 1e-10));
    }
    float newC0=0.5*(W0*newL[0]+W1*newL[1]+W2*newL[2]);
    float newC1=1.5*(W0*newL[0]*(-SQ3_5)+W2*newL[2]*SQ3_5);
    float p2_0=0.4, p2_1=-0.5, p2_2=0.4;
    float newC2=2.5*(W0*newL[0]*p2_0+W1*newL[1]*p2_1+W2*newL[2]*p2_2);

    float3 refl = reflectDir(WorldRayDirection(), fNrm);
    Payload np; initPayload(np, p.lamStart, p.lamEnd, newC0, newC1, newC2,
                            p.bounce+1, p.generation,
                            p.specularChain, p.lightRGB, rng);
    float3 Dprop = p.D + p.V * p.hitT;
    np.D = Dprop - 2.0*dot(Dprop, fNrm)*fNrm;
    np.V      = p.V      - 2.0*dot(p.V, fNrm)*fNrm;
    incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
    RayDesc nr; nr.Origin=pos+fNrm*0.001+refl*0.01; nr.Direction=refl;
    nr.TMin=0.001; nr.TMax=10000.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
    mergeChild(p, np, float3(1.0));
}

// ---- Dielectric ----
void handleDielectric(inout Payload p, float3 pos, float3 nrm, GpuMaterial mat)
{
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
        // rotSplit, rotDisp: un-propagated state → child D,V (no shift into surface).
        // Dcheck: propagated D for split condition only.
        float3 rotSplit = rotateToNewFrame(p.D, WorldRayDirection(), newDir);
        float3 rotDisp  = rotateToNewFrame(p.V, WorldRayDirection(), newDir);
        float3 Dcheck   = rotateToNewFrame(p.D + p.V * p.hitT, WorldRayDirection(), newDir);

        float cosTheta = abs(dot(WorldRayDirection(), dn));
        float r0 = (ior_mid-1.0)/(ior_mid+1.0); r0 *= r0;
        float fresnelR = r0 + (1.0-r0)*pow(1.0-cosTheta, 5.0);

        if (decayIsDead(p.throughput, p.decayC2))
            return;

        // MC importance sampling: pick reflect or refract via Fresnel probability
        bool traceReflection;
        if (fresnelR > 0.999)
            traceReflection = true;
        else if (fresnelR < 0.001)
            traceReflection = false;
        else
            traceReflection = rand1(p.rng) < fresnelR;

        if (traceReflection) {
            float3 refl = reflectDir(WorldRayDirection(), dn);
            Payload rp; initPayload(rp, p.lamStart, p.lamEnd,
                p.throughput,
                p.decayC1, p.decayC2, p.bounce+1, p.generation,
                p.specularChain, p.lightRGB, p.rng);
            float3 DpropRefl = p.D + p.V * p.hitT;
            rp.D = DpropRefl - 2.0*dot(DpropRefl, dn)*dn;
            rp.V      = p.V-2.0*dot(p.V,dn)*dn;
            RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
            rr.TMin=0.001; rr.TMax=10000.0;
            incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
            TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
            mergeChild(p, rp, float3(1.0));
        } else {
            float3 childV = rotDisp + newDisp;
            float3 Dperp = Dcheck - dot(Dcheck, newDir) * newDir;
            float d = 2.0 * length(Dperp);

            // Also check the unpropagated spread that traceSpectralSplit uses
            float3 splitDperp = rotSplit - dot(rotSplit, newDir) * newDir;
            float splitD = 2.0 * length(splitDperp);

            if (d > params.forceSplitWidth && splitD > params.forceSplitWidth &&
                params.forceSplitWidth > 0.0 &&
                p.generation < 7 &&
                (p.throughput - 0.5 * p.decayC2) > -3.0) {
                traceSpectralSplit(p, pos-dn*0.001, newDir,
                                   rotSplit, childV,
                                   p.throughput, p.decayC1, p.decayC2,
                                   p.bounce+1, p.generation+1);
            } else {
                Payload tp; initPayload(tp, p.lamStart, p.lamEnd,
                    p.throughput, p.decayC1, p.decayC2, p.bounce+1, p.generation,
                    p.specularChain, p.lightRGB, p.rng);
                tp.D = rotSplit; tp.V = rotDisp + newDisp;
                RayDesc tr; tr.Origin=pos-dn*0.001+newDir*0.01; tr.Direction=newDir;
                tr.TMin=0.001; tr.TMax=10000.0;
                incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
                TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, tr, tp);
                mergeChild(p, tp, float3(1.0));
            }
        }
    } else {
        // TIR — total internal reflection
        float3 refl = reflectDir(WorldRayDirection(), dn);
        Payload rp; initPayload(rp, p.lamStart, p.lamEnd,
            p.throughput, p.decayC1, p.decayC2, p.bounce+1, p.generation,
            p.specularChain, p.lightRGB, p.rng);
        float3 DpropTIR = p.D + p.V * p.hitT;
        rp.D = DpropTIR - 2.0*dot(DpropTIR, dn)*dn;
        rp.V      = p.V-2.0*dot(p.V, dn)*dn;
        RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
        rr.TMin=0.001; rr.TMax=10000.0;
        incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
        mergeChild(p, rp, float3(1.0));
    }
}

// ---- Photon mode helpers ----

void storePhoton(inout Payload p, float3 pos, float3 fNrm)
{
    uint slot;
    InterlockedAdd(photonCounter[0], 1, slot);
    if (slot >= MAX_PHOTONS) return;

    PhotonRecord rec;
    rec.phPos = float4(pos, 0.0);
    rec.phNrm = float4(fNrm, 0.0);
    rec.phDir = float4(normalize(WorldRayDirection()), 0.0);

    // Spectral energy: Wiener spectrum × Legendre attenuation.
    // Legendre coefficients track log-energy level + transport attenuation;
    // spectral detail comes from estimateSpectrum (matching sky's S * D pattern).
    float lamMid = float(p.lamStart + p.lamEnd) * 0.5;
    float halfW  = float(p.lamEnd - p.lamStart) * 0.5;
    float invHW  = 1.0 / max(halfW, 0.001);

    float specBucket[10];
    for (int b = 0; b < 10; b++) {
        float bS = float(LAMBDA_MIN + 40 * b);
        float bE = bS + 40.0;
        float oS = max(float(p.lamStart), bS);
        float oE = min(float(p.lamEnd), bE);
        if (oS >= oE) { specBucket[b] = 0.0; continue; }

        float lamMidB = (oS + oE) * 0.5;
        float x = clamp((lamMidB - lamMid) * invHW, -1.0, 1.0);
        float S = estimateSpectrum(lamMidB, p.lightRGB);
        float D = evalDecay(p.throughput, p.decayC1, p.decayC2, x);
        specBucket[b] = S * D * (oE - oS);
    }

    rec.spectral0_3 = float4(specBucket[0], specBucket[1], specBucket[2], specBucket[3]);
    rec.spectral4_7 = float4(specBucket[4], specBucket[5], specBucket[6], specBucket[7]);
    rec.spectral8_9 = float2(specBucket[8], specBucket[9]);
    photonBuffer[slot] = rec;
}

// 1st-order Manhattan-neighbor blend: self + 6 face-adjacent cells = 7 total.
// Constant cost, no ring expansion. Normal check handled by hash_gather.
void gatherPhotons(float3 pos, float3 nrm,
                   inout float spectral[NUM_SPECTRAL_BUCKETS])
{
    int3 cell = computeCell(pos, params.hashCellSize);

    float specAccum[NUM_SPECTRAL_BUCKETS];
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) specAccum[b] = 0.0;
    float totalWeight = 0.0;

    int3 neighbors[7] = {
        int3( 0, 0, 0),
        int3( 1, 0, 0), int3(-1, 0, 0),
        int3( 0, 1, 0), int3( 0,-1, 0),
        int3( 0, 0, 1), int3( 0, 0,-1),
    };

    for (int i = 0; i < 7; i++) {
        uint nHash = computeCellHash(cell + neighbors[i]);
        uint nBase = nHash * 10u;

        float nSum = 0.0;
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            nSum += asfloat(gatheredCellData[nBase + b]);

        if (nSum <= 0.0) continue;

        totalWeight += 1.0;
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            specAccum[b] += asfloat(gatheredCellData[nBase + b]);
    }

    if (totalWeight <= 0.0) return;

    float invW = 1.0 / totalWeight;
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        spectral[b] += specAccum[b] * invW;
}

void handlePhotonHit(inout Payload p, float3 pos, float3 fNrm, float3 nrm, int matType)
{
    if (p.bounce >= params.photonMaxBounces || decayIsDead(p.throughput, p.decayC2))
        return;

    if (p.bounce >= 3) {
        float linTp = exp(p.throughput - 0.5 * p.decayC2);
        float survive = max(linTp, 0.1);
        if (rand1(p.rng) > survive) return;
        p.throughput -= log(survive);
    }

    GpuMaterial mat = materials.items[clamp(uint(matType), 0u,
        max(0u, uint(params.materialCount) - 1u))];

    if (matType == MATERIAL_DIELECTRIC) {
        p.specularChain++;
        handleDielectric(p, pos, nrm, mat);
    }
    else if (matType == MATERIAL_METAL) {
        p.specularChain++;
        handleMetal(p, pos, fNrm, mat, p.rng);
    }
    else if (matType == MATERIAL_LAMBERTIAN || matType == MATERIAL_CHECKERBOARD) {
        storePhoton(p, pos, fNrm);
    }
}
