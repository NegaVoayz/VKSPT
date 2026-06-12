// Hash grid dispatch 1: count photons per spatial cell
import common.types;
import common.bindings;
import common.hash_utils;
import rt.payload;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= photonCounter[0]) return;

    PhotonRecord ph = photonBuffer[i];
    int3 cell = computeCell(ph.phPos.xyz, params.gatherRadius);
    uint hash = computeCellHash(cell);
    InterlockedAdd(hashCellData[hash * 2 + 0], 1);
}
