// RT pipeline material handlers — each uses Payload + TraceRay recursion
import common.types;
import common.bindings;
import geometry.normals;
import geometry.dispersion;
import math.sampling;
import math.spectral_math;
import rt.payload;
import rt.direct_light;

// ---- Checkerboard ----
void handleCheckerboard(inout Payload p, float3 pos, float3 fNrm, inout uint rng)
{
    float cx = floor(pos.x*0.5), cz = floor(pos.z*0.5);
    float3 alb = ((int(cx+cz)%2)==0) ? float3(0.75) : float3(0.15);
    p.color = exp(p.throughput) * directLight(pos, fNrm, alb, rng);

    float3 sDir = cosineSampleHemisphere(fNrm, rng);
    Payload np; initPayload(np, p.lamStart, p.lamEnd,
        p.throughput + log(dot(alb, float3(0.333))),
        p.decayC1, p.decayC2, p.bounce+1, p.generation, rng);
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
    p.color = exp(p.throughput) * directLight(pos, fNrm, alb, rng);

    float3 sDir = cosineSampleHemisphere(fNrm, rng);
    Payload np; initPayload(np, p.lamStart, p.lamEnd,
        p.throughput + log(dot(alb, float3(0.333))),
        p.decayC1, p.decayC2, p.bounce+1, p.generation, rng);
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
                            p.bounce+1, p.generation, rng);
    np.D = p.D - 2.0*dot(p.D, fNrm)*fNrm;
    np.V      = p.V      - 2.0*dot(p.V, fNrm)*fNrm;
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
        float3 rotSplit = rotateToNewFrame(p.D, WorldRayDirection(), newDir);
        float3 rotDisp  = rotateToNewFrame(p.V, WorldRayDirection(), newDir);

        float cosTheta = abs(dot(WorldRayDirection(), dn));
        float r0 = (ior_mid-1.0)/(ior_mid+1.0); r0 *= r0;
        float fresnelR = r0 + (1.0-r0)*pow(1.0-cosTheta, 5.0);

        if (fresnelR > 0.001 && !decayIsDead(p.throughput+log(fresnelR), p.decayC2)) {
            float3 refl = reflectDir(WorldRayDirection(), dn);
            Payload rp; initPayload(rp, p.lamStart, p.lamEnd,
                p.throughput+log(max(fresnelR,1e-10)),
                p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
            rp.D = p.D-2.0*dot(p.D,dn)*dn;
            rp.V      = p.V-2.0*dot(p.V,dn)*dn;
            RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
            rr.TMin=0.001; rr.TMax=10000.0;
            TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
            mergeChild(p, rp, float3(1.0));
        }

        float refrTp = p.throughput + log(max(1.0-fresnelR, 1e-10));
        if (!decayIsDead(refrTp, p.decayC2)) {
            float3 childD = rotSplit;
            float3 childV = rotDisp + newDisp;
            float3 Dperp = childD - dot(childD, newDir) * newDir;
            float d = 2.0 * length(Dperp);

            if (d > params.forceSplitWidth && params.forceSplitWidth > 0.0 &&
                p.generation < 7) {
                traceSpectralSplit(p, pos-dn*0.001, newDir,
                                   childD, childV,
                                   refrTp, p.decayC1, p.decayC2,
                                   p.bounce+1, p.generation+1);
            } else {
                Payload tp; initPayload(tp, p.lamStart, p.lamEnd,
                    refrTp, p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
                tp.D = rotSplit; tp.V = rotDisp + newDisp;
                RayDesc tr; tr.Origin=pos-dn*0.001+newDir*0.01; tr.Direction=newDir;
                tr.TMin=0.001; tr.TMax=10000.0;
                TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, tr, tp);
                mergeChild(p, tp, float3(1.0));
            }
        }
    } else {
        // TIR — total internal reflection
        float3 refl = reflectDir(WorldRayDirection(), dn);
        Payload rp; initPayload(rp, p.lamStart, p.lamEnd,
            p.throughput, p.decayC1, p.decayC2, p.bounce+1, p.generation, p.rng);
        rp.D = p.D-2.0*dot(p.D,dn)*dn;
        rp.V      = p.V-2.0*dot(p.V,dn)*dn;
        RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
        rr.TMin=0.001; rr.TMax=10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
        mergeChild(p, rp, float3(1.0));
    }
}
