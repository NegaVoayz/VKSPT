// spectral_common.slang
// Shared spectral-math definitions for VKRT shaders.
// Uses GLSL-compatible syntax (compiled via slangc).

// ---------------------------------------------------------------------------
// Wavelength domain constants
// ---------------------------------------------------------------------------
const int LAMBDA_MIN      = 380;   // nm
const int LAMBDA_MAX      = 780;   // nm (inclusive)
const int SPECTRUM_BINS   = 401;   // LAMBDA_MAX - LAMBDA_MIN + 1
const int NUM_SPECTRAL_SAMPLES = 16;

// ---------------------------------------------------------------------------
// Material type tags
// ---------------------------------------------------------------------------
const int MATERIAL_DIELECTRIC  = 0;
const int MATERIAL_METAL       = 1;
const int MATERIAL_LAMBERTIAN  = 2;

// ---------------------------------------------------------------------------
// GPU-side material (std140, 64 bytes)
// ---------------------------------------------------------------------------
struct GpuMaterial {
    vec4  cauchyA;       // A coefficients (RGB), w unused
    vec4  cauchyB;       // B coefficients (RGB), w unused   — B is given for λ in μm
    vec4  albedo;        // base colour / reflectance
    vec4  params;        // x=ior, y=roughness, z=type, w unused
};

// ---------------------------------------------------------------------------
// Cauchy formula: n(λ) = A + B / λ²
//   mat   – material with per-channel A and B
//   lambdaNm – wavelength in nanometres
// Returns per-channel IOR as vec3.
// For non-dielectric materials falls back to mat.params.x (base IOR).
// ---------------------------------------------------------------------------
vec3 cauchyIOR(GpuMaterial mat, float lambdaNm) {
    if (int(mat.params.z) != MATERIAL_DIELECTRIC) {
        return vec3(mat.params.x);
    }
    // λ in μm for the formula
    float lambdaUm  = lambdaNm * 1.0e-3;
    float lambdaUm2 = lambdaUm * lambdaUm;
    return mat.cauchyA.rgb + mat.cauchyB.rgb / lambdaUm2;
}

// ---------------------------------------------------------------------------
// Snell's law refraction.
//   I     – incident direction (normalised, pointing toward surface)
//   N     – surface normal (normalised, pointing toward incident ray)
//   eta   – ratio n_incoming / n_transmitted
// Returns the refracted direction.
// On total internal reflection returns vec3(0) and sets a sentinel flag
// via the `out bool` parameter, or returns the reflected direction.
// ---------------------------------------------------------------------------
bool refractDir(vec3 I, vec3 N, float eta, out vec3 T) {
    float cosI   = -dot(I, N);          // cosine of incident angle
    float sinT2  = eta * eta * (1.0 - cosI * cosI);

    if (sinT2 > 1.0) {
        T = vec3(0.0);                   // total internal reflection — no transmission
        return false;
    }

    float cosT = sqrt(1.0 - sinT2);
    T = eta * I + (eta * cosI - cosT) * N;
    return true;
}

// ---------------------------------------------------------------------------
// Reflect direction around a normal.
// ---------------------------------------------------------------------------
vec3 reflectDir(vec3 I, vec3 N) {
    return I - 2.0 * dot(I, N) * N;
}

// ---------------------------------------------------------------------------
// Wavelength → RGB  (design doc Appendix B)
//   Piecewise spectrum with violet/red falloff attenuation.
// ---------------------------------------------------------------------------
vec3 wavelengthToRGB(float lambdaNm) {
    vec3 rgb = vec3(0.0);

    if (lambdaNm < 440.0) {
        rgb = vec3(0.5 - 0.5 * (lambdaNm - 380.0) / 60.0, 0.0, 1.0);
    }
    else if (lambdaNm < 490.0) {
        rgb = vec3(0.0, (lambdaNm - 440.0) / 50.0, 1.0);
    }
    else if (lambdaNm < 510.0) {
        rgb = vec3(0.0, 1.0, 1.0 - (lambdaNm - 490.0) / 20.0);
    }
    else if (lambdaNm < 580.0) {
        rgb = vec3((lambdaNm - 510.0) / 70.0, 1.0, 0.0);
    }
    else if (lambdaNm < 645.0) {
        rgb = vec3(1.0, 1.0 - (lambdaNm - 580.0) / 65.0, 0.0);
    }
    else if (lambdaNm <= 780.0) {
        rgb = vec3(1.0, 0.0, 0.0);
    }

    // Attenuation at the spectral ends (human eye sensitivity falloff)
    float attenuation = 1.0;
    if (lambdaNm < 420.0) {
        attenuation = 0.3 + 0.7 * (lambdaNm - 380.0) / 40.0;
    }
    if (lambdaNm > 700.0) {
        attenuation = 0.3 + 0.7 * (780.0 - lambdaNm) / 80.0;
    }
    return rgb * attenuation;
}

// ---------------------------------------------------------------------------
// RGB → dominant wavelength (crude inverse)
//   Used to estimate the "native" wavelength of a coloured surface
//   for spectral absorption.
// ---------------------------------------------------------------------------
float rgbToDominantWavelength(vec3 rgb) {
    float r = rgb.r;
    float g = rgb.g;
    float b = rgb.b;

    // Find the dominant channel
    if (r >= g && r >= b) {
        // Red-dominant: map hue from orange (~610) to deep red (~700)
        float purity = (r - max(g, b));
        return mix(610.0, 680.0, clamp(purity, 0.0, 1.0));
    }
    else if (g >= r && g >= b) {
        // Green-dominant
        float purity = (g - max(r, b));
        return mix(510.0, 555.0, clamp(purity, 0.0, 1.0));
    }
    else {
        // Blue-dominant
        float purity = (b - max(r, g));
        return mix(430.0, 475.0, clamp(purity, 0.0, 1.0));
    }
}

// ---------------------------------------------------------------------------
// Spectral absorption for coloured Lambertian surfaces.
//   Attenuates wavelengths that are far from the surface's native colour.
//   Uses a Gaussian falloff around the dominant wavelength.
// ---------------------------------------------------------------------------
float spectralAbsorption(float lambdaNm, vec3 surfaceAlbedo) {
    float nativeLambda = rgbToDominantWavelength(surfaceAlbedo);

    // Falloff sigma: ~50 nm (narrow enough to separate colours,
    // wide enough to avoid hard clipping)
    float sigma = 50.0;
    float diff  = lambdaNm - nativeLambda;
    float atten = exp(-0.5 * (diff * diff) / (sigma * sigma));

    // Scale by luminance so dark surfaces absorb more overall
    float lum = dot(surfaceAlbedo, vec3(0.299, 0.587, 0.114));
    return atten * (0.3 + 0.7 * lum);
}
