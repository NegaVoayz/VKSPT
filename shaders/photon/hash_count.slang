// Hash grid dispatch 1: count photons per spatial cell.
// On first batch, also zeros gatheredCellData for cross-batch atomic accumulation.
import common.types;
import common.bindings;
import common.hash_utils;
import rt.payload;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // Thread 0 zeros gatheredCellData before first batch accumulates into it
    if (tid.x == 0 && params.batchIndex == 0) {
        for (uint k = 0; k < HASH_TABLE_SIZE * 10u; k++)
            gatheredCellData[k] = 0u;
    }

    uint i = tid.x;
    if (i >= photonCounter[0]) return;

    PhotonRecord ph = photonBuffer[i];
    int3 cell = computeCell(ph.phPos.xyz, params.hashCellSize);
    uint hash = computeCellHash(cell);
    InterlockedAdd(hashCellData[hash * 2 + 0], 1);
}
