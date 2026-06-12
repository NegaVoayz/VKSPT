// Shared spatial hash functions for photon hash grid (count / scatter / gather)
import common.types;

// Spatial hash: maps 3D cell coordinates to [0, HASH_TABLE_SIZE)
uint computeCellHash(int3 cell)
{
    uint h = (uint(cell.x) * 73856093u)
           ^ (uint(cell.y) * 19349663u)
           ^ (uint(cell.z) * 83492791u);
    return h & (uint(HASH_TABLE_SIZE) - 1u);
}

// Compute cell coordinates from a world-space position
int3 computeCell(float3 pos, float cellSize)
{
    return int3(floor(pos / cellSize));
}
