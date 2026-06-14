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
//   output[b] = I_t × num[b] / denom / (π·cellSz²)
// π·cellSz² = cell cross-section — converts specSum → irradiance density (fixed norm)
//
// Irradiance I_t is NOT smoothed — luminance edges stay sharp.
// Normal check (dot < 0.2) prevents cross-geometry-edge color bleed.
// Outlier kill: cells with cnt < minGatherRadius are suppressed — camera trace
//   fallback finds the nearest reliable neighbor at render time.
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
    uint outBase = hash * 10u;

    if (cnt == 0) return;

    // Outlier suppression: cells with too few photons have unreliable spectra.
    // Skip writing gatheredCellData — camera trace fallback finds nearest reliable cell.
    if (cnt < uint(params.minGatherRadius)) return;

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
    }}}

    // Self-cell is included in neighbor loop: nHash==hash, d=0, normal self-match.

    if (denom <= 0.0) return;

    // Kernel area π·cellSz² converts specSum → spectral irradiance density.
    // Fixed to cell cross-section — independent of sigma so brightness stays
    // constant when tuning smoothing radius.
    float kernelArea = PI * cellSz * cellSz;
    float invDenom = 1.0 / (denom * kernelArea + 1e-10);
    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        atomicAddFloat(outBase + b, I_t * num[b] * invDenom);
}
