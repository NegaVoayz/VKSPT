// Photon gather: 1st-order Manhattan-neighbor blend with I/S separation.
// Called by camera-path material handlers (checkerboard / lambertian).
// Hash-collision guard: stored centerPos must belong to the queried spatial cell.
import common.types;
import common.bindings;
import common.hash_utils;
import rt.payload;

// 1st-order Manhattan-neighbor blend: self + 6 face-adjacent cells = 7 total.
// displayCellData: 13 uint/cell = 10 spectral + 3 position (asfloat).
void gatherPhotons(float3 pos, float3 nrm,
                   inout float spectral[NUM_SPECTRAL_BUCKETS])
{
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

        totalWeight += 1.0;
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            specAccum[b] += asfloat(displayCellData[nBase + b]);
    }

    if (totalWeight <= 0.0) return;

    float invW = 1.0 / totalWeight;
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        spectral[b] += specAccum[b] * invW;
}
