// Direct lighting with shadow rays — imports bindings for tlas, lights
import common.types;
import common.bindings;

[ForceInline]
public float3 sampleDirectLighting(float3 pos, float3 nrm, float3 albedo, uint rng, float diffuseStrength, int numLights) {
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
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        float3 shadowOrg = pos + nrm * 0.001 + toLight * 0.01;
        RayDesc shadowRay = { shadowOrg, 0.001, toLight, tMax };
        rq.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, shadowRay);
        while (rq.Proceed()) {}
        if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            float r2 = distToLight * distToLight + 1.0;
            float att = lintensity / max(r2, 1.0);
            if (ltype == LIGHT_DIRECTIONAL) att = lintensity;
            float3 diffContrib = lcolor * (albedo * 0.31831) * (att * nDotL * spotAtt);
            contrib += diffContrib * diffuseStrength;
        }
    }
    return contrib;
}
