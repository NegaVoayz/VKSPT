// Color science: sRGB gamma, CMF lookup, spectrum estimation, sky integration
import common.cmf_data;
import math.spectral_math;

public float3 srgbToLinear(float3 c) {
    float3 lo = c / 12.92;
    float3 hi = pow((c + 0.055) / 1.055, float3(2.4));
    return lerp(lo, hi, step(float3(0.04045), c));
}

public float3 linearToSRGB(float3 c) {
    float3 lo = c * 12.92;
    float3 hi = pow(c, float3(1.0/2.4)) * 1.055 - 0.055;
    return lerp(lo, hi, step(float3(0.0031308), c));
}

public float3 displayCMF(float lamNm) {
    float t  = (lamNm - 380.0) / 5.0;
    int   i0 = clamp(int(floor(t)), 0, 80);
    int   i1 = clamp(i0 + 1, 0, 80);
    float f  = frac(t);
    float3 v0 = float3(CMF_R[i0], CMF_G[i0], CMF_B[i0]);
    float3 v1 = float3(CMF_R[i1], CMF_G[i1], CMF_B[i1]);
    return lerp(v0, v1, f);
}

public float3 spectrumSampleToSRGB(float lamNm) {
    float3 lin = displayCMF(lamNm);
    return linearToSRGB(lin);
}

public float estimateSpectrum(float lamNm, float3 linearRGB) {
    float t  = (lamNm - 380.0) / 5.0;
    int   i0 = clamp(int(floor(t)), 0, 80);
    int   i1 = clamp(i0 + 1, 0, 80);
    float f  = frac(t);
    float s0 = dot(linearRGB, float3(BASIS_R[i0], BASIS_G[i0], BASIS_B[i0]));
    float s1 = dot(linearRGB, float3(BASIS_R[i1], BASIS_G[i1], BASIS_B[i1]));
    return lerp(s0, s1, f);
}

public float3 integrateSkySRGB(float lamMid, float halfWidth,
                       float c0, float c1, float c2,
                       float3 envLinear) {
    float bandWidth = halfWidth * 2.0;
    if (bandWidth > 350.0 && abs(c1) < 0.001 && abs(c2) < 0.001) {
        return envLinear * c0;
    }
    const float XK[7] = { -0.94910791, -0.74153119, -0.40584515, 0.0, 0.40584515, 0.74153119, 0.94910791 };
    const float WK[7] = { 0.12948497, 0.27970539, 0.38183005, 0.41795918, 0.38183005, 0.27970539, 0.12948497 };
    float3 linSum = float3(0.0);
    for (int i = 0; i < 7; i++) {
        float lam_k  = lamMid + halfWidth * XK[i];
        float S_k    = estimateSpectrum(lam_k, envLinear);
        float D_k    = evalDecay(c0, c1, c2, XK[i]);
        float3 cmf_k  = displayCMF(lam_k);
        linSum += WK[i] * S_k * D_k * cmf_k * halfWidth;
    }
    return linSum;
}

public float3 wavelengthToRGB(float lam) { return spectrumSampleToSRGB(lam); }

public float rgbToDomLambda(float3 rgb) {
    if (rgb.r >= rgb.g && rgb.r >= rgb.b) return lerp(610.0, 680.0, clamp(rgb.r-max(rgb.g,rgb.b),0.0,1.0));
    else if (rgb.g >= rgb.r && rgb.g >= rgb.b) return lerp(510.0, 555.0, clamp(rgb.g-max(rgb.r,rgb.b),0.0,1.0));
    else return lerp(430.0, 475.0, clamp(rgb.b-max(rgb.r,rgb.g),0.0,1.0));
}
