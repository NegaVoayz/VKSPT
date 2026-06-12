// Photon light source sampling + RGB spectrum → Legendre projection
import common.types;
import common.bindings;
import common.cmf_data;
import math.color;
import math.spectral_math;
import math.sampling;
import rt.payload;

// Weighted random light selection by intensity.
int selectLightByIntensity(int numLights, inout uint rng)
{
    float totalIntensity = 0.0;
    float intensities[8];
    int n = min(numLights, 8);
    for (int i = 0; i < n; i++) {
        intensities[i] = lights.items[i].colorIntensity.w;
        totalIntensity += intensities[i];
    }
    if (totalIntensity <= 0.0) return 0;

    float r = rand1(rng) * totalIntensity;
    float accum = 0.0;
    for (int j = 0; j < n; j++) {
        accum += intensities[j];
        if (r <= accum) return j;
    }
    return n - 1;
}

// Sample origin and direction on a light source.
void sampleLightSource(GpuLight l, out float3 origin, out float3 dir, inout uint rng)
{
    int lt = int(l.posType.w);

    if (lt == LIGHT_DIRECTIONAL) {
        float3 lightDir = normalize(l.posType.xyz);
        float3 u, v;
        orthonormalBasis(lightDir, u, v);
        float r = sqrt(rand1(rng)) * 100.0;
        float theta = rand1(rng) * 6.283185307;
        // Sample a disk perpendicular to light direction, placed far back.
        origin = params.camOrigin
               + u * (r * cos(theta)) + v * (r * sin(theta))
               - lightDir * 2000.0;
        dir = lightDir;
    }
    else if (lt == LIGHT_POINT) {
        origin = l.posType.xyz;
        dir = uniformSampleSphere(rng);
    }
    else if (lt == LIGHT_SPOT) {
        origin = l.posType.xyz;
        float cosHalf = cos(l.outerRange.x * 3.14159265 / 180.0);
        dir = sampleCone(l.dirInner.xyz, cosHalf, rng);
    }
    else { // ambient or unknown — sample sphere from origin
        origin = params.camOrigin;
        dir = uniformSampleSphere(rng);
    }
}

// RGB → full spectrum reconstruction (Wiener basis) → log-Legendre projection.
// Uses estimateSpectrum at 3 Gauss-Legendre points on [LAMBDA_MIN, LAMBDA_MAX],
// takes ln(), fits 2nd-order Legendre coefficients.
// RGB → Legendre: photon starts like camera ray (c0=c1=c2=0).
// Attenuation accumulates during transport; light spectrum applied at storage via estimateSpectrum.
void rgbSpectrumToLegendre(float3 rgb, out float c0, out float c1, out float c2)
{
    c0 = 0.0;
    c1 = 0.0;
    c2 = 0.0;
}
