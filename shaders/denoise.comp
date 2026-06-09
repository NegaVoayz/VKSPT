#version 460

// =============================================================================
// Phase 6.5: Light edge-avoiding spatial denoiser
//
// 2-scale filter (step 1, 2) with 3×3 kernel. Tight edge-stopping
// weights preserve surface detail. Cross-frame accumulation handles
// most of the denoising — this is just a subtle cleanup pass.
// Sky pixels (normal ≈ 0) pass through unfiltered.
//
// Input:  accumBuffer  (binding 13) — per-pixel float4 (r,g,b,sampleCount)
//         normalImage  (binding 14) — world-space normals packed [0,1]
//         depthImage   (binding 15) — primary hit distance
// Output: outputImage  (binding 1)  — denoised RGBA8
// =============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba8, binding = 1) uniform image2D outputImage;
layout(set = 0, binding = 13, std430) buffer AccumBlock { float data[]; } accumData;
layout(rgba16f, binding = 14) uniform image2D normalImage;
layout(r32f, binding = 15) uniform image2D depthImage;

float normalWeight(vec3 na, vec3 nb) {
    if (length(na) < 0.001 || length(nb) < 0.001) return 1.0;
    float d2 = dot(na - nb, na - nb);
    return exp(-d2 / 0.02);   // σ² = 0.02 → σ ≈ 0.14
}

float depthWeight(float da, float db) {
    float diff = abs(da - db);
    return exp(-diff * diff / 0.1);  // σ² = 0.1 → σ ≈ 0.32
}

float colorWeight(vec3 ca, vec3 cb) {
    float la = dot(ca, vec3(0.299, 0.587, 0.114));
    float lb = dot(cb, vec3(0.299, 0.587, 0.114));
    float d2 = (la - lb) * (la - lb);
    return exp(-d2 / 0.02);   // σ² = 0.02 → σ ≈ 0.14
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(outputImage);
    if (px.x >= sz.x || px.y >= sz.y) return;

    uint idx = uint(px.y * sz.x + px.x);
    uint base = idx * 4u;
    vec3 centerColor = vec3(accumData.data[base], accumData.data[base+1], accumData.data[base+2]);
    float centerCount = accumData.data[base+3];
    if (centerCount > 0.0) centerColor /= centerCount;

    vec3 centerNormal = imageLoad(normalImage, px).xyz * 2.0 - 1.0;
    float centerDepth = imageLoad(depthImage, px).x;

    // Sky pixel → pass through unfiltered
    if (length(centerNormal) < 0.001) {
        imageStore(outputImage, px, vec4(centerColor, 1.0));
        return;
    }

    // ---- 2 scales: step=1 (fine), step=2 (coarse) ----
    // Heavily weighted toward fine scale (0.8) to preserve detail
    vec3 result = vec3(0.0);
    float totalW = 0.0;

    for (int iter = 0; iter < 2; iter++) {
        int step = 1 + iter;  // 1, 2
        float scaleW = (iter == 0) ? 0.8 : 0.2;

        vec3 sum = vec3(0.0);
        float wsum = 0.0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ivec2 qp = px + ivec2(dx * step, dy * step);
                if (qp.x < 0 || qp.x >= sz.x || qp.y < 0 || qp.y >= sz.y) continue;

                uint qi = uint(qp.y * sz.x + qp.x);
                uint qb = qi * 4u;
                vec3 qColor = vec3(accumData.data[qb], accumData.data[qb+1], accumData.data[qb+2]);
                float qCount = accumData.data[qb+3];
                if (qCount > 0.0) qColor /= qCount;

                vec3 qNormal = imageLoad(normalImage, qp).xyz * 2.0 - 1.0;
                float qDepth = imageLoad(depthImage, qp).x;

                if (length(qNormal) < 0.001) continue;  // skip sky neighbors

                float w = normalWeight(centerNormal, qNormal)
                        * depthWeight(centerDepth, qDepth)
                        * colorWeight(centerColor, qColor);
                sum += qColor * w;
                wsum += w;
            }
        }

        if (wsum > 0.0) {
            result += scaleW * (sum / wsum);
            totalW += scaleW;
        }
    }

    if (totalW > 0.0) {
        result /= totalW;
    } else {
        result = centerColor;
    }

    result = max(result, vec3(0.0));
    imageStore(outputImage, px, vec4(result, 1.0));
}
