// Hash grid dispatch 2: exclusive prefix sum of per-cell counts → offsets
// Single workgroup of 1024 threads, each processing 256 entries
import common.types;
import common.bindings;

static const uint ELEMS_PER_THREAD = 256;

groupshared uint sharedPrefix[1024];

[shader("compute")]
[numthreads(1024, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint t = tid.x;
    uint base = t * ELEMS_PER_THREAD;

    // Pass 1: compute local sum (no register array — single accumulator)
    uint localSum = 0;
    for (uint j = 0; j < ELEMS_PER_THREAD; j++)
        localSum += hashCellData[(base + j) * 2 + 0];

    sharedPrefix[t] = localSum;
    GroupMemoryBarrierWithGroupSync();

    // Thread 0 does single-threaded exclusive scan of 1024 partial sums
    if (t == 0) {
        uint accum = 0;
        for (uint k = 0; k < 1024; k++) {
            uint val = sharedPrefix[k];
            sharedPrefix[k] = accum;
            accum += val;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint blockPrefix = sharedPrefix[t];

    // Pass 2: re-read counts, write offsets + zero counts
    uint runningSum = blockPrefix;
    for (uint j = 0; j < ELEMS_PER_THREAD; j++) {
        uint idx = base + j;
        uint val = hashCellData[idx * 2 + 0];
        hashCellData[idx * 2 + 1] = runningSum;
        hashCellData[idx * 2 + 0] = 0;
        runningSum += val;
    }
}
