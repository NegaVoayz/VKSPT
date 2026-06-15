// 2-scale filter (step 1, 2) with 3×3 kernel. Tight edge-stopping
// weights preserve surface detail. Cross-frame accumulation handles
// most of the denoising — this is just a subtle cleanup pass.
// Sky pixels (normal ≈ 0) pass through unfiltered.
[[vk::binding(1, 0)]]  RWTexture2D<float4> outputImage;
[[vk::binding(13, 0)]] RWStructuredBuffer<float> accumData;
[[vk::binding(14, 0)]] RWTexture2D<float4> normalImage;
[[vk::binding(15, 0)]] RWTexture2D<float>  depthImage;

float normalWeight(float3 na, float3 nb) {
    if (length(na) < 0.001 || length(nb) < 0.001) return 1.0;
    float d2 = dot(na - nb, na - nb);
    return exp(-d2 / 0.02);   // σ² = 0.02 → σ ≈ 0.14
}

float depthWeight(float da, float db) {
    float diff = abs(da - db);
    return exp(-diff * diff / 0.1);  // σ² = 0.1 → σ ≈ 0.32
}

float colorWeight(float3 ca, float3 cb) {
    float la = dot(ca, float3(0.299, 0.587, 0.114));
    float lb = dot(cb, float3(0.299, 0.587, 0.114));
    float d2 = (la - lb) * (la - lb);
    return exp(-d2 / 0.02);   // σ² = 0.02 → σ ≈ 0.14
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID) {
    int2 px = int2(dispatchID.xy);
    int w, h;
    outputImage.GetDimensions(w, h);
    int2 sz = int2(w, h);
    if (px.x >= sz.x || px.y >= sz.y) return;

    uint idx = uint(px.y * sz.x + px.x);
    uint base = idx * 4u;
    float3 centerColor = float3(accumData[base], accumData[base+1], accumData[base+2]);
    float centerCount = accumData[base+3];
    if (centerCount > 0.0) centerColor /= centerCount;

    float3 centerNormal = normalImage.Load(int2(px.x, px.y)).xyz * 2.0 - 1.0;
    float centerDepth = depthImage.Load(int2(px.x, px.y)).x;

    if (length(centerNormal) < 0.001) {
        outputImage[int2(px.x, px.y)] = float4(centerColor, 1.0);
        return;
    }

    // Heavily weighted toward fine scale (0.8) to preserve detail
    float3 result = float3(0.0);
    float totalW = 0.0;

    for (int iter = 0; iter < 2; iter++) {
        int step = 1 + iter;  // 1, 2
        float scaleW = (iter == 0) ? 0.8 : 0.2;

        float3 sum = float3(0.0);
        float wsum = 0.0;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int2 qp = px + int2(dx * step, dy * step);
                if (qp.x < 0 || qp.x >= sz.x || qp.y < 0 || qp.y >= sz.y) continue;

                uint qi = uint(qp.y * sz.x + qp.x);
                uint qb = qi * 4u;
                float3 qColor = float3(accumData[qb], accumData[qb+1], accumData[qb+2]);
                float qCount = accumData[qb+3];
                if (qCount > 0.0) qColor /= qCount;

                float3 qNormal = normalImage.Load(int2(qp.x, qp.y)).xyz * 2.0 - 1.0;
                float qDepth = depthImage.Load(int2(qp.x, qp.y)).x;

                if (length(qNormal) < 0.001) continue;

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

    result = max(result, float3(0.0));
    outputImage[int2(px.x, px.y)] = float4(result, 1.0);
}
