// Hash grid dispatch 5: per-cell spectral gather with count-based spread radius.
// Each neighbor's photon count determines its spread radius: r = maxR / sqrt(N).
// More photons → smaller radius → sharper reconstruction (higher confidence).
// Uses 1/(d²+ε²) kernel with r² decay for natural energy spreading on 2D surfaces.
// Gather (pull) architecture: each target cell pulls from neighbors within the
// neighbor's spread radius — one write per thread, no within-batch atomics.
// Cross-batch accumulation via CAS-loop atomic float add.
//
// Reads  cellPhotonData (23 floats/cell from hash_aggregate)
// Writes gatheredCellData (10 uint/cell — atomic-accumulated, asfloat on read)
import common.types;
import common.bindings;
import common.hash_utils;
import rt.payload;

static const uint FLOATS_PER_CELL = 23;
static const float PI = 3.14159265;

// CAS-loop float atomic add on gatheredCellData (declared as uint).
// Sequential dispatches across batches → zero contention → first-try success.
void atomicAddFloat(uint idx, float val)
{
    uint prev = gatheredCellData[idx];
    uint cur, orig;
    do {
        cur = prev;
        float sum = asfloat(cur) + val;
        InterlockedCompareExchange(gatheredCellData[idx], cur, asuint(sum), orig);
        prev = orig;
    } while (prev != cur);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint hash = tid.x;
    if (hash >= HASH_TABLE_SIZE) return;

    uint cnt    = hashCellData[hash * 2 + 0];
    uint inBase = hash * FLOATS_PER_CELL;
    uint outBase = hash * 10u;

    if (cnt == 0) return;

    float3 centerPos = float3(cellPhotonData[inBase + 0],
                              cellPhotonData[inBase + 1],
                              cellPhotonData[inBase + 2]);
    float3 centerNrm = float3(cellPhotonData[inBase + 4],
                              cellPhotonData[inBase + 5],
                              cellPhotonData[inBase + 6]);

    float specOut[10] = {0,0,0,0,0,0,0,0,0,0};

    float cellSz = params.hashCellSize;
    float maxR   = params.maxGatherRadius;
    float minR   = params.minGatherRadius;
    float eps    = cellSz;
    float eps2   = eps * eps;

    int3 cMin = computeCell(centerPos - maxR, cellSz);
    int3 cMax = computeCell(centerPos + maxR, cellSz);

    for (int cx = cMin.x; cx <= cMax.x; cx++) {
    for (int cy = cMin.y; cy <= cMax.y; cy++) {
    for (int cz = cMin.z; cz <= cMax.z; cz++) {
        uint nHash = computeCellHash(int3(cx, cy, cz));
        uint nBase = nHash * FLOATS_PER_CELL;
        float nCnt = cellPhotonData[nBase + 22];
        if (nCnt == 0.0) continue;

        // Neighbor's spread radius from its photon count
        float rNbr  = clamp(maxR / sqrt(nCnt), minR, maxR);
        float rNbr2 = rNbr * rNbr;

        float3 nPos = float3(cellPhotonData[nBase + 0],
                             cellPhotonData[nBase + 1],
                             cellPhotonData[nBase + 2]);
        float3 d = nPos - centerPos;
        float d2 = dot(d, d);
        if (d2 >= rNbr2) continue;

        float3 nNrm = float3(cellPhotonData[nBase + 4],
                             cellPhotonData[nBase + 5],
                             cellPhotonData[nBase + 6]);
        if (dot(nNrm, centerNrm) < 0.2) continue;

        // Normalized 1/(d²+ε²) kernel
        // Integral over disk radius rNbr: π·ln((rNbr²+ε²)/ε²)
        float w    = 1.0 / (d2 + eps2);
        float norm = PI * log((rNbr2 + eps2) / eps2);
        float wn   = w / max(norm, 1e-10);

        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            specOut[b] += cellPhotonData[nBase + 12 + b] * wn;
    }}}

    // Self-cell is included in neighbor loop: nHash==hash, d=0, normal self-match.

    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        atomicAddFloat(outBase + b, specOut[b]);
}
