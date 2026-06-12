// RT pipeline direct lighting — shadow rays via TraceRay
import common.types;
import common.bindings;
import rt.payload;

float3 directLight(float3 pos, float3 nrm, float3 albedo, inout uint rng)
{
    float3 contrib = float3(0.0);
    for (int li = 0; li < params.numLights; li++) {
        GpuLight l = lights.items[li];
        int lt = int(l.posType.w);
        float3 toLight; float tM, att, spA = 1.0;
        if (lt == LIGHT_DIRECTIONAL) {
            toLight = l.posType.xyz; tM = 10000.0;
            att = l.colorIntensity.w * 0.01;
        } else {
            toLight = l.posType.xyz - pos; float d = length(toLight);
            toLight /= d; tM = d - 0.001;
            att = l.colorIntensity.w / max(d*d, 1.0);
            if (lt == LIGHT_SPOT) {
                float cA = dot(-toLight, l.dirInner.xyz);
                spA = smoothstep(l.outerRange.x, l.outerRange.x+0.05, cA);
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
            float3 lc = l.colorIntensity.xyz;
            contrib += lc * (albedo * 0.31831) * (att * nDL * spA) * params.diffuseStrength;
        }
    }
    return contrib;
}
