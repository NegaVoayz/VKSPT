// Surface + volume absorption models
import common.types;
import math.color;

public float spectralAbsorption(float lam, float3 albedo) {
    float nl  = rgbToDomLambda(albedo);
    float d   = lam - nl;
    float att = exp(-0.5 * d * d / 7200.0);
    float lum = dot(albedo, float3(0.299, 0.587, 0.114));
    float maxC = max(max(albedo.r, albedo.g), albedo.b);
    float minC = min(min(albedo.r, albedo.g), albedo.b);
    float sat  = (maxC - minC) / max(maxC, 0.001);
    float attMix = lerp(1.0, att, sat);
    return attMix * (0.07 + 0.93 * lum);
}

public float computeAbsorption(GpuMaterial m, float lamNm) {
    float lu = lamNm * 1e-3, l2 = lu * lu;
    return m.absorpA.y + m.absorpB.y / l2;
}
