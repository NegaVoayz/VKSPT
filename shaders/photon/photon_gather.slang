// Photon gather: 3×3×3 Gaussian-weighted spatial filter.
// Called by camera-path material handlers (checkerboard / lambertian).
// Hash-collision guard: stored centerPos must belong to the queried spatial cell.
import common.types;
import common.bindings;
import common.hash_utils;
import rt.payload;

// Manhattan-neighbor blend with Gaussian weights: self + 6 face-adjacent = 7 cells.
// σ = 1.0, weight = exp(-distSq / (2σ²)). Center weight=1.0, face weight≈0.607.
// displayCellData: 13 uint/cell = 10 spectral + 3 position (asfloat).
void gatherPhotons(float3 pos, float3 nrm,
                   inout float spectral[NUM_SPECTRAL_BUCKETS])
{
    static const float INV_TWO_SIGMA_SQ = 0.5; // 1/(2*1.0²)
    static const float GAUSS_FACE = exp(-1.0 * INV_TWO_SIGMA_SQ);

    float cellSz = params.hashCellSize;
    int3 cell = computeCell(pos, cellSz);

    float specAccum[NUM_SPECTRAL_BUCKETS];
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) specAccum[b] = 0.0;
    float totalWeight = 0.0;

    int3 neighbors[7] = {
        int3( 0, 0, 0),
        int3( 1, 0, 0), int3(-1, 0, 0),
        int3( 0, 1, 0), int3( 0,-1, 0),
        int3( 0, 0, 1), int3( 0, 0,-1),
    };
    float gaussWeights[7] = { 1.0,
        GAUSS_FACE, GAUSS_FACE, GAUSS_FACE, GAUSS_FACE, GAUSS_FACE, GAUSS_FACE };

    for (int i = 0; i < 7; i++) {
        int3 qCell = cell + neighbors[i];
        uint nHash = computeCellHash(qCell);
        uint nBase = nHash * 13u;

        // Hash-collision guard: stored position must be in the queried cell
        float3 phCenter = float3(asfloat(displayCellData[nBase + 10]),
                                  asfloat(displayCellData[nBase + 11]),
                                  asfloat(displayCellData[nBase + 12]));
        if (any(computeCell(phCenter, cellSz) != qCell)) continue;

        float nSum = 0.0;
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            nSum += asfloat(displayCellData[nBase + b]);

        if (nSum <= 0.0) continue;

        float w = gaussWeights[i];
        totalWeight += w;
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            specAccum[b] += asfloat(displayCellData[nBase + b]) * w;
    }

    if (totalWeight <= 0.0) return;

    float invW = 1.0 / totalWeight;
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        spectral[b] += specAccum[b] * invW;
}
