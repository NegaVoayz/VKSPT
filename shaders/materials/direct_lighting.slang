// Direct lighting with shadow rays + spectral caustics through dielectrics
import common.types;
import common.bindings;
import math.spectral_math;
import math.color;
import geometry.normals;

[ForceInline]
public float3 sampleDirectLighting(float3 pos, float3 nrm, float3 albedo, uint rng, float diffuseStrength, int numLights, inout float spectral[NUM_SPECTRAL_BUCKETS]) {
    float3 contrib = float3(0.0);
    for (int li = 0; li < numLights; li++) {
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

        // Shadow ray from surface towards light
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq0;
        float3 sOrg = pos + nrm * 0.001 + toLight * 0.01;
        RayDesc sRay0 = { sOrg, 0.001, toLight, tMax };
        rq0.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, sRay0);
        while (rq0.Proceed()) {}

        if (rq0.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            // Direct path: no hit → light reaches surface
            float r2 = distToLight * distToLight + 1.0;
            float att = lintensity / max(r2, 1.0);
            if (ltype == LIGHT_DIRECTIONAL) att = lintensity * 0.01;
            contrib += lcolor * (albedo * 0.31831) * (att * nDotL * spotAtt) * diffuseStrength;
            continue;
        }

        float tEntry = rq0.CommittedRayT();
        int miEntry = rq0.CommittedInstanceID();
        int primEntry = rq0.CommittedPrimitiveIndex();
        uint matIdxEntry = instanceRanges[0].materialID[miEntry];
        GpuMaterial entryMat = materials.items[clamp(matIdxEntry, 0u, 15u)];

        if (int(entryMat.params.z) != MATERIAL_DIELECTRIC) {
            // Opaque object blocks light
            continue;
        }

        // ---- Spectral caustics: trace through dielectric per bucket ----
        float3 entryNrm = computeGeometricNormal(miEntry, primEntry);
        // Flip so normal faces the incoming ray direction
        if (dot(entryNrm, toLight) >= 0.0) entryNrm = -entryNrm;

        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
            float lam = float(LAMBDA_MIN + 40 * b + 20); // bucket center nm
            float ior = cauchyIOR(entryMat, lam);
            float etaEnter = 1.0 / ior;

            float3 insideDir;
            if (!refractDir(toLight, entryNrm, etaEnter, insideDir))
                continue; // TIR at entry — this wavelength doesn't pass

            // Trace through the dielectric interior
            float3 insideOrg = sOrg + toLight * tEntry + insideDir * 0.01;
            float insideTMax = tMax - tEntry - 0.01;
            RayQuery<RAY_FLAG_FORCE_OPAQUE> rq1;
            RayDesc insideRay = { insideOrg, 0.001, insideDir, insideTMax };
            rq1.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, insideRay);
            while (rq1.Proceed()) {}

            if (rq1.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
                continue; // no exit face hit (shouldn't happen for closed mesh)
            int miExit = rq1.CommittedInstanceID();
            if (miExit != miEntry) continue; // hit different object

            int primExit = rq1.CommittedPrimitiveIndex();
            float tInside = rq1.CommittedRayT();
            float3 exitNrm = computeGeometricNormal(miExit, primExit);
            // Flip so normal faces the incoming ray direction (from inside)
            if (dot(exitNrm, insideDir) >= 0.0) exitNrm = -exitNrm;

            float3 finalDir;
            if (!refractDir(insideDir, exitNrm, ior, finalDir))
                continue; // TIR at exit

            // Trace from exit point towards the light
            float3 exitOrg = insideOrg + insideDir * tInside + finalDir * 0.01;
            float exitTMax = tMax - tEntry - tInside - 0.02;
            float3 toLightFromExit = ltype == LIGHT_DIRECTIONAL ? toLight : normalize(lpos - exitOrg);

            RayQuery<RAY_FLAG_FORCE_OPAQUE> rq2;
            RayDesc exitRay = { exitOrg, 0.001, toLightFromExit, exitTMax };
            rq2.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, exitRay);
            while (rq2.Proceed()) {}

            if (rq2.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                continue; // something blocks the light after the prism

            // Light reached! Accumulate into spectral bucket
            float r2 = distToLight * distToLight + 1.0;
            float att = lintensity / max(r2, 1.0);
            if (ltype == LIGHT_DIRECTIONAL) att = lintensity * 0.01;
            float S_lam = estimateSpectrum(lam, lcolor);
            float surfRefl = estimateSpectrum(lam, albedo);
            spectral[b] += S_lam * surfRefl * att * diffuseStrength * 40.0;
        }
    }
    return contrib;
}
