#version 460
#extension GL_EXT_ray_query : require

// Pull in spectral math + material definitions
#include "spectral_common.slang"

// ---------------------------------------------------------------------------
// Descriptor bindings (set = 0)
// ---------------------------------------------------------------------------
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba8) uniform image2D       outputImage;
layout(set = 0, binding = 2) uniform MaterialBlock {
    GpuMaterial items[8];   // max 8 materials
} materials;

// ---------------------------------------------------------------------------
// Push constants
//   std430: vec3 = 12 bytes but padded to 16-byte alignment.
//   Total 96 bytes.
// ---------------------------------------------------------------------------
layout(push_constant) uniform PushConstants {
    vec3  camOrigin;
    float _pad0;
    vec3  camU;
    float _pad1;
    vec3  camV;
    float _pad2;
    vec3  camW;
    float _pad3;
    int   samplesPerPixel;
    int   maxBounces;
    int   materialCount;
    float _pad4;
} params;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// ---------------------------------------------------------------------------
// Sky — acts as a white emitter behind the prism.
// Spectral rays passing through the prism keep full energy and disperse;
// the white background lets the wavelength-dependent refraction produce
// visible chromatic separation.
// ---------------------------------------------------------------------------
vec3 skyEmission(vec3 direction) {
    // Soft gradient for visual orientation (sky → ground hint),
    // but always near-white so the prism dispersion is clearly visible.
    float t = 0.5 * (direction.y + 1.0);
    return mix(vec3(0.85, 0.88, 0.92), vec3(1.0, 1.0, 1.0), t);
}

// ---------------------------------------------------------------------------
// Lambertian shade at a hit point (monochrome-like for coloured surfaces)
// ---------------------------------------------------------------------------
vec3 lambertianShade(vec3 normal, vec3 lightDir, vec3 lightColor, vec3 albedo) {
    float NdotL  = max(dot(normal, lightDir), 0.0);
    vec3 ambient = vec3(0.08, 0.08, 0.10);
    return albedo * (ambient + lightColor * NdotL);
}

// ---------------------------------------------------------------------------
// Ray generation: map pixel to direction through the image plane
// ---------------------------------------------------------------------------
vec3 generateRayDirection(ivec2 pixel, ivec2 imageSize) {
    vec2 uv = (vec2(pixel) + 0.5) / vec2(imageSize) * 2.0 - 1.0;
    uv.y = -uv.y;   // flip Y (image coords top-left, NDC bottom-left)
    return normalize(params.camW + uv.x * params.camU + uv.y * params.camV);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
void main() {
    ivec2 pixel   = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imgSize = imageSize(outputImage);

    if (pixel.x >= imgSize.x || pixel.y >= imgSize.y) {
        return;
    }

    // ---- Discrete spectrum accumulator (per-nm bins) ----
    vec3 spectrum[SPECTRUM_BINS];
    for (int i = 0; i < SPECTRUM_BINS; ++i) {
        spectrum[i] = vec3(0.0);
    }

    vec3 rayOrigin = params.camOrigin;
    vec3 rayDir    = generateRayDirection(pixel, imgSize);

    // Fixed directional light for lambertian surfaces
    vec3 lightDir   = normalize(vec3(1.0, 2.0, -1.0));
    vec3 lightColor = vec3(1.0, 0.95, 0.9);

    int numSamples = max(params.samplesPerPixel, 1);

    // ---- Spectral sample loop ----
    for (int si = 0; si < numSamples; ++si) {
        // Uniformly-spaced wavelengths across [380, 780]
        float lambdaBinF = float(LAMBDA_MIN) +
            (float(LAMBDA_MAX) - float(LAMBDA_MIN)) *
            (float(si) + 0.5) / float(numSamples);
        int lambdaBin = int(lambdaBinF);
        lambdaBin = clamp(lambdaBin, LAMBDA_MIN, LAMBDA_MAX);

        vec3 energy = vec3(1.0);

        // ===== Bounce 0: camera → scene =====
        rayQueryEXT rq0;
        rayQueryInitializeEXT(rq0, tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                              rayOrigin, 0.001, rayDir, 10000.0);
        while (rayQueryProceedEXT(rq0)) {}

        if (rayQueryGetIntersectionTypeEXT(rq0, true) !=
            gl_RayQueryCommittedIntersectionTriangleEXT)
        {
            // Miss — sky
            energy *= skyEmission(rayDir);
            spectrum[lambdaBin - LAMBDA_MIN] += energy;
            continue;
        }

        // Hit 0 — identify the face via instance custom index
        float  t0       = rayQueryGetIntersectionTEXT(rq0, true);
        vec3   pos0     = rayOrigin + rayDir * t0;
        int    matIdx0  = rayQueryGetIntersectionInstanceCustomIndexEXT(rq0, true);
        vec3   objNrm0  = vec3(0.0, 0.0, 1.0);  // all triangles face +Z in object-space
        mat4x3 o2w0     = rayQueryGetIntersectionObjectToWorldEXT(rq0, true);
        mat3   wm0      = mat3(o2w0);
        vec3   nrm0     = normalize(transpose(inverse(wm0)) * objNrm0);
        if (dot(nrm0, -rayDir) < 0.0) nrm0 = -nrm0;  // face the ray

        GpuMaterial mat0 = materials.items[clamp(matIdx0, 0, 7)];

        // ---- Bounce 0 action ----
        vec3 outOrigin;
        vec3 outDir;
        int  outMatIdx;

        if (int(mat0.params.z) != MATERIAL_DIELECTRIC) {
            // Non-dielectric — shade and deposit directly
            if (int(mat0.params.z) == MATERIAL_LAMBERTIAN) {
                float att = spectralAbsorption(float(lambdaBin), mat0.albedo.rgb);
                energy *= att * lambertianShade(nrm0, lightDir, lightColor,
                                                mat0.albedo.rgb);
            }
            spectrum[lambdaBin - LAMBDA_MIN] += energy;
            continue;
        }

        // Dielectric: compute refraction for this wavelength
        float iorEntry = cauchyIOR(mat0, float(lambdaBin)).g;  // green-channel IOR
        float eta0 = 1.0 / iorEntry;   // air → glass

        vec3 refractDir0;
        bool refrOk0 = refractDir(rayDir, nrm0, eta0, refractDir0);
        if (!refrOk0) {
            // Total internal reflection — shouldn't happen on entry, but be safe
            outDir = reflectDir(rayDir, nrm0);
        } else {
            outDir = refractDir0;
        }
        outOrigin = pos0 + outDir * 0.01;

        // ===== Bounce 1: inside glass → hit exit face =====
        rayQueryEXT rq1;
        rayQueryInitializeEXT(rq1, tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                              outOrigin, 0.001, outDir, 10000.0);
        while (rayQueryProceedEXT(rq1)) {}

        if (rayQueryGetIntersectionTypeEXT(rq1, true) !=
            gl_RayQueryCommittedIntersectionTriangleEXT)
        {
            // Exited prism without hitting back face — sky
            energy *= skyEmission(outDir);
            spectrum[lambdaBin - LAMBDA_MIN] += energy;
            continue;
        }

        float t1   = rayQueryGetIntersectionTEXT(rq1, true);
        vec3  pos1 = outOrigin + outDir * t1;
        int   matIdx1 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq1, true);
        vec3  objNrm1 = vec3(0.0, 0.0, 1.0);
        mat4x3 o2w1   = rayQueryGetIntersectionObjectToWorldEXT(rq1, true);
        mat3   wm1    = mat3(o2w1);
        vec3   nrm1   = normalize(transpose(inverse(wm1)) * objNrm1);
        if (dot(nrm1, -outDir) < 0.0) nrm1 = -nrm1;

        GpuMaterial mat1 = materials.items[clamp(matIdx1, 0, 7)];

        // ---- Bounce 1 action ----
        if (int(mat1.params.z) != MATERIAL_DIELECTRIC) {
            if (int(mat1.params.z) == MATERIAL_LAMBERTIAN) {
                float att = spectralAbsorption(float(lambdaBin), mat1.albedo.rgb);
                energy *= att * lambertianShade(nrm1, lightDir, lightColor,
                                                mat1.albedo.rgb);
            }
            spectrum[lambdaBin - LAMBDA_MIN] += energy;
            continue;
        }

        // Refract out: glass → air
        float iorExit = cauchyIOR(mat1, float(lambdaBin)).g;
        float eta1 = iorExit / 1.0;   // glass → air

        // nrm1 was already flipped by the face-toward-ray check above.
        // It points into the glass (toward the incident ray coming from inside).
        // refractDir expects N to point toward the incident medium — correct as-is.
        vec3 refractDir1;
        bool refrOk1 = refractDir(outDir, nrm1, eta1, refractDir1);
        if (!refrOk1) {
            // TIR inside the prism — reflect internally
            outDir = reflectDir(outDir, nrm1);
        } else {
            outDir = refractDir1;
        }
        outOrigin = pos1 + outDir * 0.01;

        // ===== Bounce 2: in air → sky or ground =====
        rayQueryEXT rq2;
        rayQueryInitializeEXT(rq2, tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                              outOrigin, 0.001, outDir, 10000.0);
        while (rayQueryProceedEXT(rq2)) {}

        if (rayQueryGetIntersectionTypeEXT(rq2, true) !=
            gl_RayQueryCommittedIntersectionTriangleEXT)
        {
            // Hit nothing — sky
            energy *= skyEmission(outDir);
            spectrum[lambdaBin - LAMBDA_MIN] += energy;
            continue;
        }

        float t2   = rayQueryGetIntersectionTEXT(rq2, true);
        vec3  pos2 = outOrigin + outDir * t2;
        int   matIdx2 = rayQueryGetIntersectionInstanceCustomIndexEXT(rq2, true);
        vec3  objNrm2 = vec3(0.0, 0.0, 1.0);
        mat4x3 o2w2   = rayQueryGetIntersectionObjectToWorldEXT(rq2, true);
        mat3   wm2    = mat3(o2w2);
        vec3   nrm2   = normalize(transpose(inverse(wm2)) * objNrm2);
        if (dot(nrm2, -outDir) < 0.0) nrm2 = -nrm2;

        GpuMaterial mat2 = materials.items[clamp(matIdx2, 0, 7)];

        if (int(mat2.params.z) == MATERIAL_LAMBERTIAN) {
            float att = spectralAbsorption(float(lambdaBin), mat2.albedo.rgb);
            energy *= att * lambertianShade(nrm2, lightDir, lightColor,
                                            mat2.albedo.rgb);
        } else {
            energy *= skyEmission(outDir);
        }

        spectrum[lambdaBin - LAMBDA_MIN] += energy;
    }

    // ---- Convert accumulated spectrum to RGB ----
    vec3 color = vec3(0.0);
    for (int lam = LAMBDA_MIN; lam <= LAMBDA_MAX; ++lam) {
        vec3 specVal = spectrum[lam - LAMBDA_MIN];
        if (specVal.r > 0.0 || specVal.g > 0.0 || specVal.b > 0.0) {
            color += wavelengthToRGB(float(lam)) * specVal;
        }
    }
    color /= float(numSamples);

    imageStore(outputImage, pixel, vec4(color, 1.0));
}
