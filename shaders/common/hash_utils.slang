// Shared spatial hash functions for photon hash grid (count / scatter / gather)
import common.types;

// Morton 3D + high-bit mixing: interleave bits for spatial locality,
// then finalize with avalanche mixing for uniform bucket distribution.
uint computeCellHash(int3 cell)
{
    uint x = uint(cell.x);
    uint y = uint(cell.y);
    uint z = uint(cell.z);

    // Morton 3D interleave
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;

    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8))  & 0x0300F00F;
    y = (y | (y << 4))  & 0x030C30C3;
    y = (y | (y << 2))  & 0x09249249;

    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z << 8))  & 0x0300F00F;
    z = (z | (z << 4))  & 0x030C30C3;
    z = (z | (z << 2))  & 0x09249249;

    uint morton = x | (y << 1) | (z << 2);

    // High-bit finalizer — avalanche mixing
    morton ^= morton >> 16;
    morton *= 0x85ebca6bu;
    morton ^= morton >> 13;
    morton *= 0xc2b2ae35u;
    morton ^= morton >> 16;

    return morton & (uint(HASH_TABLE_SIZE) - 1u);
}

// Compute cell coordinates from a world-space position
int3 computeCell(float3 pos, float cellSize)
{
    return int3(floor(pos / cellSize));
}
