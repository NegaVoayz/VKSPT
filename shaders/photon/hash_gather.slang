// Hash grid dispatch 5: per-cell spectral irradiance output.
// Direct per-cell data — no spatial filter. Temporal denoising is done
// by the blend_photon EMA on displayCellData.
// Pass-normalized via params.passCount.  CAS-loop atomic float add.
//
// Reads  cellPhotonData (23 floats/cell from hash_aggregate)
// Writes gatheredCellData (13 uint/cell — 10 spectral + 3 position)
import common.types;
import common.bindings;
import rt.payload;

static const float PI = 3.14159265;

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
    uint inBase = hash * 23u;
    uint outBase = hash * 13u;

    if (cnt == 0) return;

    float cellSz  = params.hashCellSize;
    float kernelArea = 0.5 * PI * cellSz * cellSz;
    float invPass = 1.0 / max(float(params.passCount), 1.0);
    float invKA = invPass / kernelArea;

    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        atomicAddFloat(outBase + b, cellPhotonData[inBase + 12 + b] * invKA);

    gatheredCellData[outBase + 10] = asuint(cellPhotonData[inBase + 0]);
    gatheredCellData[outBase + 11] = asuint(cellPhotonData[inBase + 1]);
    gatheredCellData[outBase + 12] = asuint(cellPhotonData[inBase + 2]);
}
