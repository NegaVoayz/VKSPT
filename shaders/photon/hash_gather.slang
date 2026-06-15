// Hash grid dispatch 5: per-cell spectral gather with irradiance/spectrum separation.
//
// Each cell's output = irradiance × smoothed_spectrum:
//   I_t   = Σ_b specSum_t[b]                        — target irradiance (kept sharp)
//   S_j   = specSum_j[b] / Φ_j                       — neighbor normalized spectrum
//   w_j   = Φ_j × exp(-d²/(2σ²))                     — energy-weighted spatial Gaussian
//   Ŝ     = Σ(w_j × S_j) / Σ w_j                     — smoothed spectrum
//   output = I_t × Ŝ                                 — recombine: sharp brightness × smooth color
//
// Equivalent single-pass form (Φ_j cancels in w_j × S_j):
//   num[b] = Σ_j G(d) × specSum_j[b]
//   denom  = Σ_j Φ_j × G(d)
//   output[b] = I_t × num[b] / denom / (0.5·π·cellSz²)
// 0.5·π·cellSz² = half cell cross-section — compensates for fixed vs count-based norm
//
// Neighborhood density: G-weighted avg photon count ≥ minNeighborPhotons required.
// Irradiance I_t is NOT smoothed — luminance edges stay sharp.
// Outlier dampening: if I_t > 2× G-weighted neighbor average, clamped to 2×.
//   Prevents single-cell fireflies while preserving real brightness edges.
// Normal check (dot < 0.2) prevents cross-geometry-edge color bleed.
// Cross-batch accumulation via CAS-loop atomic float add on gatheredCellData.
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
    uint outBase = hash * 13u;   // 10 spectral + 3 position (hash-collision guard)

    if (cnt == 0) return;

    float3 centerPos = float3(cellPhotonData[inBase + 0],
                              cellPhotonData[inBase + 1],
                              cellPhotonData[inBase + 2]);
    float3 centerNrm = float3(cellPhotonData[inBase + 4],
                              cellPhotonData[inBase + 5],
                              cellPhotonData[inBase + 6]);

    // Target irradiance: sum of all spectral buckets (kept sharp, not smoothed)
    float I_t = 0.0;
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        I_t += cellPhotonData[inBase + 12 + b];
    if (I_t <= 0.0) return;

    float cellSz  = params.hashCellSize;
    float searchR = params.maxGatherRadius;
    float sigma   = searchR / 2.0;                // broad σ: heavy neighbor weighting
    float invTwoSigma2 = 1.0 / (2.0 * sigma * sigma + 1e-10);

    float num[NUM_SPECTRAL_BUCKETS];
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) num[b] = 0.0;
    float denom = 0.0;
    float G_sum = 0.0;   // sum of spatial Gaussian weights for consistency check
    float G_cnt = 0.0;   // sum of G × nCnt for neighborhood density

    int3 cMin = computeCell(centerPos - searchR, cellSz);
    int3 cMax = computeCell(centerPos + searchR, cellSz);

    for (int cx = cMin.x; cx <= cMax.x; cx++) {
    for (int cy = cMin.y; cy <= cMax.y; cy++) {
    for (int cz = cMin.z; cz <= cMax.z; cz++) {
        uint nHash = computeCellHash(int3(cx, cy, cz));
        uint nBase = nHash * FLOATS_PER_CELL;
        float nCnt = cellPhotonData[nBase + 22];
        if (nCnt == 0.0) continue;

        float3 nPos = float3(cellPhotonData[nBase + 0],
                             cellPhotonData[nBase + 1],
                             cellPhotonData[nBase + 2]);

        // Hash-collision guard: stored avg position must belong to the queried cell
        if (any(computeCell(nPos, cellSz) != int3(cx, cy, cz))) continue;

        float3 d = nPos - centerPos;
        float d2 = dot(d, d);

        // Spatial Gaussian weight
        float G = exp(-d2 * invTwoSigma2);

        float3 nNrm = float3(cellPhotonData[nBase + 4],
                             cellPhotonData[nBase + 5],
                             cellPhotonData[nBase + 6]);
        if (dot(nNrm, centerNrm) < 0.2) continue;

        // Φ_j = Σ_b specSum_j[b] (reliability weight), accumulate num in single pass
        float Phi_j = 0.0;
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
            float spec = cellPhotonData[nBase + 12 + b];
            Phi_j += spec;
            num[b] += G * spec;
        }
        if (Phi_j <= 0.0) continue;

        denom += Phi_j * G;   // energy-weighted spatial weight
        G_sum += G;            // spatial weight for outlier detection
        G_cnt += G * nCnt;     // weighted photon count for neighborhood density
    }}}

    // Self-cell is included in neighbor loop: nHash==hash, d=0, normal self-match.

    if (denom <= 0.0) return;

    // Neighborhood density: G-weighted average neighbor photon count
    float nbrAvgCnt = G_cnt / max(G_sum, 1e-10);
    if (nbrAvgCnt < params.minNeighborPhotons) return;  // 邻域太稀疏 → 放弃

    // Isolated-cell kill: neighbors have no photons → current cell is noise
    float neighborDenom = denom - I_t;  // denom减去自己的贡献(I_t × G(0)=I_t)
    if (neighborDenom < I_t * 0.5) return;  // 邻域有效能量 < 自身的50% → 放弃

    // Local consistency: dampen irradiance outliers
    // If I_t >> neighbor average, pull it down to avoid single-cell fireflies.
    // G-weighted mean neighbor energy: denom / G_sum = Σ(Φ_j×G) / Σ G
    static const float OUTLIER_MAX_RATIO = 2.0;
    float I_nbr_avg = denom / max(G_sum, 1e-10);
    if (I_t > I_nbr_avg * OUTLIER_MAX_RATIO)
        I_t = I_nbr_avg * OUTLIER_MAX_RATIO;

    // Kernel area π·cellSz² converts specSum → spectral irradiance density.
    // Fixed to cell cross-section — independent of sigma so brightness stays
    // constant when tuning smoothing radius.
    float kernelArea = 0.5 * PI * cellSz * cellSz;
    float invDenom = 1.0 / (denom * kernelArea + 1e-10);
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        atomicAddFloat(outBase + b, I_t * num[b] * invDenom);

    // Store cell center for hash-collision guard in camera trace
    gatheredCellData[outBase + 10] = asuint(centerPos.x);
    gatheredCellData[outBase + 11] = asuint(centerPos.y);
    gatheredCellData[outBase + 12] = asuint(centerPos.z);
}
