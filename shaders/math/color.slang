// Color science: sRGB gamma, CMF lookup, spectrum estimation, sky integration

vec3 srgbToLinear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 linearToSRGB(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = pow(c, vec3(1.0/2.4)) * 1.055 - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 displayCMF(float lamNm) {
    float t  = (lamNm - 380.0) / 5.0;
    int   i0 = clamp(int(floor(t)), 0, 80);
    int   i1 = clamp(i0 + 1, 0, 80);
    float f  = fract(t);
    vec3 v0 = vec3(CMF_R[i0], CMF_G[i0], CMF_B[i0]);
    vec3 v1 = vec3(CMF_R[i1], CMF_G[i1], CMF_B[i1]);
    return mix(v0, v1, f);
}

vec3 spectrumSampleToSRGB(float lamNm) {
    vec3 lin = displayCMF(lamNm);
    return linearToSRGB(lin);
}

float estimateSpectrum(float lamNm, vec3 linearRGB) {
    float t  = (lamNm - 380.0) / 5.0;
    int   i0 = clamp(int(floor(t)), 0, 80);
    int   i1 = clamp(i0 + 1, 0, 80);
    float f  = fract(t);
    float s0 = dot(linearRGB, vec3(BASIS_R[i0], BASIS_G[i0], BASIS_B[i0]));
    float s1 = dot(linearRGB, vec3(BASIS_R[i1], BASIS_G[i1], BASIS_B[i1]));
    return mix(s0, s1, f);
}

vec3 integrateSkySRGB(float lamMid, float halfWidth,
                       float c0, float c1, float c2,
                       vec3 envLinear) {
    float bandWidth = halfWidth * 2.0;
    if (bandWidth > 350.0 && abs(c1) < 0.001 && abs(c2) < 0.001) {
        return envLinear * c0;
    }
    const float XK[7] = { -0.94910791, -0.74153119, -0.40584515, 0.0, 0.40584515, 0.74153119, 0.94910791 };
    const float WK[7] = { 0.12948497, 0.27970539, 0.38183005, 0.41795918, 0.38183005, 0.27970539, 0.12948497 };
    vec3 linSum = vec3(0.0);
    for (int i = 0; i < 7; i++) {
        float lam_k  = lamMid + halfWidth * XK[i];
        float S_k    = estimateSpectrum(lam_k, envLinear);
        float D_k    = evalDecay(c0, c1, c2, XK[i]);
        vec3  cmf_k  = displayCMF(lam_k);
        linSum += WK[i] * S_k * D_k * cmf_k * halfWidth;
    }
    return linSum;
}

vec3 wavelengthToRGB(float lam) { return spectrumSampleToSRGB(lam); }

float rgbToDomLambda(vec3 rgb) {
    if (rgb.r >= rgb.g && rgb.r >= rgb.b) return mix(610.0, 680.0, clamp(rgb.r-max(rgb.g,rgb.b),0.0,1.0));
    else if (rgb.g >= rgb.r && rgb.g >= rgb.b) return mix(510.0, 555.0, clamp(rgb.g-max(rgb.r,rgb.b),0.0,1.0));
    else return mix(430.0, 475.0, clamp(rgb.b-max(rgb.r,rgb.g),0.0,1.0));
}
