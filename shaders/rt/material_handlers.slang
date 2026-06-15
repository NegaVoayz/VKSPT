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
import photon.photon_gather;

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
    // p.D already propagated by apply_diff in ClosestHitMain
    np.D = p.D - 2.0*dot(p.D, fNrm)*fNrm;
    np.V = p.V - 2.0*dot(p.V, fNrm)*fNrm;
    incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
    RayDesc nr; nr.Origin=pos+fNrm*0.001+refl*0.01; nr.Direction=refl;
    nr.TMin=0.001; nr.TMax=10000.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
    mergeChild(p, np, float3(1.0));
}

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
        // D,V already propagated by apply_diff in ClosestHitMain
        float3 rotD = rotateToNewFrame(p.D, WorldRayDirection(), newDir);
        float3 rotV = rotateToNewFrame(p.V, WorldRayDirection(), newDir);
        float3 childV = rotV + newDisp;

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
            rp.D = p.D - 2.0*dot(p.D, dn)*dn;
            rp.V = p.V - 2.0*dot(p.V, dn)*dn;
            RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
            rr.TMin=0.001; rr.TMax=10000.0;
            incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
            TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
            mergeChild(p, rp, float3(1.0));
        } else {
            Payload tp; initPayload(tp, p.lamStart, p.lamEnd,
                p.throughput, p.decayC1, p.decayC2, p.bounce+1, p.generation,
                p.specularChain, p.lightRGB, p.rng);
            tp.D = rotD; tp.V = childV;
            RayDesc tr; tr.Origin=pos-dn*0.001+newDir*0.01; tr.Direction=newDir;
            tr.TMin=0.001; tr.TMax=10000.0;
            incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
            TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, tr, tp);
            mergeChild(p, tp, float3(1.0));
        }
    } else {
        // TIR — total internal reflection
        float3 refl = reflectDir(WorldRayDirection(), dn);
        Payload rp; initPayload(rp, p.lamStart, p.lamEnd,
            p.throughput, p.decayC1, p.decayC2, p.bounce+1, p.generation,
            p.specularChain, p.lightRGB, p.rng);
        rp.D = p.D - 2.0*dot(p.D, dn)*dn;
        rp.V = p.V - 2.0*dot(p.V, dn)*dn;
        RayDesc rr; rr.Origin=pos+dn*0.001+refl*0.01; rr.Direction=refl;
        rr.TMin=0.001; rr.TMax=10000.0;
        incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, rr, rp);
        mergeChild(p, rp, float3(1.0));
    }
}
