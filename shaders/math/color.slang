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
        return envLinear * exp(c0);  // log-space c0 → linear D=exp(c0)
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
    if (rgb.r >= rgb.g && rgb.r >= rgb.b)
        return lerp(610.0, 680.0, clamp(rgb.r-max(rgb.g,rgb.b),0.0,1.0));
    else if (rgb.g >= rgb.r && rgb.g >= rgb.b)
        return lerp(510.0, 555.0, clamp(rgb.g-max(rgb.r,rgb.b),0.0,1.0));
    else
        return lerp(430.0, 475.0, clamp(rgb.b-max(rgb.r,rgb.g),0.0,1.0));
}

// --- Unified absorption operators (log-domain Legendre) ---

// Expand albedo RGB to log-domain Legendre via 3-point GL quadrature.
// Gray (r==g==b) albedos skip the RGB→spectrum fit and use a flat spectrum
// to avoid green bias from estimateSpectrum's basis functions.
public void albedoLegendreExpand(float3 alb, float lamMid, float halfW,
                                  out float C0, out float C1, out float C2)
{
    if (alb.r == alb.g && alb.g == alb.b) {
        C0 = log(max(alb.r, 1e-10));
        C1 = 0.0;
        C2 = 0.0;
        return;
    }
    const float SQ3_5 = 0.77459667;
    const float W0 = 5.0/9.0, W1 = 8.0/9.0, W2 = 5.0/9.0;
    float L[3];
    for (int k = 0; k < 3; k++) {
        float xk = (k==0)?-SQ3_5:((k==1)?0.0:SQ3_5);
        L[k] = log(max(estimateSpectrum(lamMid+halfW*xk, alb), 1e-10));
    }
    C0 = 0.5*(W0*L[0]+W1*L[1]+W2*L[2]);
    C1 = 1.5*(W0*L[0]*(-SQ3_5)+W2*L[2]*SQ3_5);
    const float P2_0=0.4, P2_1=-0.5, P2_2=0.4;
    C2 = 2.5*(W0*L[0]*P2_0+W1*L[1]*P2_1+W2*L[2]*P2_2);
}

// Expand linear Legendre reflectance to log-domain Legendre in ray's frame.
public void reflectanceLegendreExpand(float r0, float r1, float r2,
                                       float lamC, float lamH,
                                       float lamMid, float halfW,
                                       out float C0, out float C1, out float C2)
{
    const float SQ3_5 = 0.77459667;
    const float W0 = 5.0/9.0, W1 = 8.0/9.0, W2 = 5.0/9.0;
    float L[3];
    for (int k = 0; k < 3; k++) {
        float xk = (k==0)?-SQ3_5:((k==1)?0.0:SQ3_5);
        float lam_k = lamMid + halfW * xk;
        float Rk = evalLegendre(r0, r1, r2, (lam_k-lamC)/lamH);
        L[k] = log(max(Rk, 1e-10));
    }
    C0 = 0.5*(W0*L[0]+W1*L[1]+W2*L[2]);
    C1 = 1.5*(W0*L[0]*(-SQ3_5)+W2*L[2]*SQ3_5);
    const float P2_0=0.4, P2_1=-0.5, P2_2=0.4;
    C2 = 2.5*(W0*L[0]*P2_0+W1*L[1]*P2_1+W2*L[2]*P2_2);
}

// Apply reflectance Legendre to light's spectral state.
// Evaluates light+reflectance at 3 GL points, reconstructs new Legendre.
public void applyAbsorption(inout int lamStart, inout int lamEnd,
                             inout float C0, inout float C1, inout float C2,
                             float reflC0, float reflC1, float reflC2)
{
    float lamMid = float(lamStart+lamEnd)*0.5;
    float halfW  = float(lamEnd-lamStart)*0.5;
    const float SQ3_5=0.77459667;
    const float W0=5.0/9.0, W1=8.0/9.0, W2=5.0/9.0;
    float newL[3];
    for (int k=0;k<3;k++){
        float xk = (k==0)?-SQ3_5:((k==1)?0.0:SQ3_5);
        float Lk = C0+C1*xk+C2*(1.5*xk*xk-0.5);
        float Rk = reflC0+reflC1*xk+reflC2*(1.5*xk*xk-0.5);
        newL[k] = Lk+Rk;
    }
    C0=0.5*(W0*newL[0]+W1*newL[1]+W2*newL[2]);
    C1=1.5*(W0*newL[0]*(-SQ3_5)+W2*newL[2]*SQ3_5);
    const float P2_0=0.4,P2_1=-0.5,P2_2=0.4;
    C2=2.5*(W0*newL[0]*P2_0+W1*newL[1]*P2_1+W2*newL[2]*P2_2);
    if (halfW<=0.0) return;
    float invHW=1.0/halfW;
    while (lamStart<lamEnd){
        float x=float(lamStart-int(lamMid))*invHW;
        x=clamp(x,-1.0,1.0);
        if ((C0+C1*x+C2*(1.5*x*x-0.5))>=-13.81551) break;
        lamStart+=max(1,(lamEnd-lamStart)/8);
    }
    while (lamEnd>lamStart){
        float x=float(lamEnd-int(lamMid))*invHW;
        x=clamp(x,-1.0,1.0);
        if ((C0+C1*x+C2*(1.5*x*x-0.5))>=-13.81551) break;
        lamEnd-=max(1,(lamEnd-lamStart)/8);
    }
}
