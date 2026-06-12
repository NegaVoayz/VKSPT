// Hash grid dispatch 3: scatter photon indices into sorted array by cell hash
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

    uint localIdx;
    InterlockedAdd(hashCellData[hash * 2 + 0], 1, localIdx);
    uint writePos = hashCellData[hash * 2 + 1] + localIdx;
    sortedPhotonIndices[writePos] = i;
}
