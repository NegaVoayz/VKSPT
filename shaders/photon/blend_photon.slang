// EMA temporal blend: smooths gatheredCellData → displayCellData every frame.
// Runs once per camera frame before the RT dispatch.
//
// Reads:  gatheredCellData (binding 19) — accumulated photon gather result
// Writes: displayCellData  (binding 20) — smoothed output consumed by photon_gather
import common.types;
import common.bindings;

static const float BLEND_ALPHA = 0.01;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint hash = tid.x;
    if (hash >= HASH_TABLE_SIZE) return;

    uint base = hash * 13u;   // 10 spectral + 3 position

    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++) {
        uint idx = base + b;
        float oldVal = asfloat(displayCellData[idx]);
        float newVal = asfloat(gatheredCellData[idx]);
        displayCellData[idx] = asuint(lerp(oldVal, newVal, BLEND_ALPHA));
    }

    // Position guard: direct copy (no smoothing needed for hash-collision check)
    displayCellData[base + 10] = gatheredCellData[base + 10];
    displayCellData[base + 11] = gatheredCellData[base + 11];
    displayCellData[base + 12] = gatheredCellData[base + 12];
}
