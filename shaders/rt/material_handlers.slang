// RT pipeline material handlers — each uses Payload + TraceRay recursion.
// Absorption is applied in ClosestHitMain BEFORE dispatch — handlers receive
// the already-absorbed payload state and just continue tracing.
import common.types;
import common.cmf_data;
import common.bindings;
import geometry.normals;
import geometry.dispersion;
import math.sampling;
import math.spectral_math;
import math.legendre_decay;
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
        p.throughput, p.decayC1, p.decayC2,
        p.bounce+1, p.generation, p.specularChain, p.lightRGB, rng);
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

    gatherPhotons(pos, fNrm, p.spectral);

    float3 sDir = cosineSampleHemisphere(fNrm, rng);
    Payload np; initPayload(np, p.lamStart, p.lamEnd,
        p.throughput, p.decayC1, p.decayC2,
        p.bounce+1, p.generation, p.specularChain, p.lightRGB, rng);
    incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
    RayDesc nr; nr.Origin=pos+fNrm*0.001+sDir*0.01; nr.Direction=sDir;
    nr.TMin=0.001; nr.TMax=10000.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
    mergeChild(p, np, alb);
}

void handleMetal(inout Payload p, float3 pos, float3 fNrm,
                  GpuMaterial mat, inout uint rng)
{
    float3 refl = reflectDir(WorldRayDirection(), fNrm);
    Payload np; initPayload(np, p.lamStart, p.lamEnd,
        p.throughput, p.decayC1, p.decayC2,
        p.bounce+1, p.generation, p.specularChain, p.lightRGB, rng);
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

    float ci = -dot(WorldRayDirection(), dn);
    float sint2 = eta_mid * eta_mid * (1.0 - ci * ci);

    // Volume absorption: Beer-Lambert through glass, applied when exiting.
    // α(λ) = A + B/λ², scalar Cauchy model (xyz→luminance since same in all channels).
    if (!entering) {
        float absA = dot(mat.absorpA.xyz, 1.0/3.0);
        float absB = dot(mat.absorpB.xyz, 1.0/3.0);
        if (absA > 0.0 || absB > 0.0)
            applyVolumeDecay(p.throughput, p.decayC1, p.decayC2,
                             float(p.lamStart+p.lamEnd)*0.5,
                             float(p.lamEnd-p.lamStart)*0.5,
                             absA, absB, p.hitT);
    }

    // Fresnel reflectance (Schlick), TIR → R=1.0
    float R0 = (ior_mid - 1.0) / (ior_mid + 1.0);
    R0 = R0 * R0;
    float fresnelR = (sint2 >= 1.0) ? 1.0
                     : (R0 + (1.0 - R0) * pow(1.0 - ci, 5.0));

    fresnelR = clamp(fresnelR * mat.params.y, 0.0, 1.0);

    if (rand1(p.rng) < fresnelR) {
        float3 refl = reflectDir(WorldRayDirection(), dn);
        Payload np; initPayload(np, p.lamStart, p.lamEnd,
            p.throughput, p.decayC1, p.decayC2,
            p.bounce+1, p.generation, p.specularChain+1, p.lightRGB, p.rng);
        np.D = p.D - 2.0*dot(p.D, dn)*dn;
        np.V = p.V - 2.0*dot(p.V, dn)*dn;
        incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
        RayDesc nr; nr.Origin=pos+dn*0.001+refl*0.01; nr.Direction=refl;
        nr.TMin=0.001; nr.TMax=10000.0;
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
        mergeChild(p, np, float3(1.0));
        return;
    }

    // Refract
    float3 T; refractDir(WorldRayDirection(), dn, eta_mid, T);

    // Spectral split for dispersion
    float tau_start = entering ? (1.0/ior_start) : ior_start;
    float tau_end   = entering ? (1.0/ior_end)   : ior_end;
    float dTau = abs(tau_end - tau_start);
    float3 splitD = p.D;
    float3 splitV = p.V + dTau * normalize(T - dot(T, dn)*dn);

    Payload np; initPayload(np, p.lamStart, p.lamEnd,
        p.throughput, p.decayC1, p.decayC2,
        p.bounce+1, p.generation, p.specularChain+1, p.lightRGB, p.rng);
    np.D = splitD - dot(splitD, T) * T;
    np.V = splitV - dot(splitV, T) * T;
    incChildRay(p.generation); recBounce(uint(p.bounce) + 1u);
    RayDesc nr; nr.Origin=pos+dn*0.001+T*0.01; nr.Direction=T;
    nr.TMin=0.001; nr.TMax=10000.0;
    TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, nr, np);
    mergeChild(p, np, float3(1.0));
}
