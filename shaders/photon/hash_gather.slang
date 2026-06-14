// Hash grid dispatch 5: per-cell spectral gather with shell-expansion.
// For each occupied cell, performs shell-expansion gather centered on the
// cell's representative position.  Edge detection uses tangent-plane dipole
// projection to avoid false triggers from 3D surface curvature.
// Output is area-normalized: divided by kernel integral (π·r²/3 for linear weight).
// Result goes to gatheredCellData for O(1) camera lookup.
//
// Reads  cellPhotonData (23 floats/cell from hash_aggregate)
// Writes gatheredCellData (10 floats/cell — area-normalized, pre-gathered spectral)
import common.types;
import common.bindings;
import common.hash_utils;
import rt.payload;

static const uint FLOATS_PER_CELL = 23;
static const float PI = 3.14159265;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint hash = tid.x;
    if (hash >= HASH_TABLE_SIZE) return;

    uint cnt   = hashCellData[hash * 2 + 0];
    uint inBase  = hash * FLOATS_PER_CELL;
    uint outBase = hash * 10u;

    if (cnt == 0) {
        for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
            gatheredCellData[outBase + b] = 0.0;
        return;
    }

    float3 centerPos = float3(cellPhotonData[inBase + 0],
                               cellPhotonData[inBase + 1],
                               cellPhotonData[inBase + 2]);
    float3 centerNrm = float3(cellPhotonData[inBase + 4],
                               cellPhotonData[inBase + 5],
                               cellPhotonData[inBase + 6]);

    float cellSz = params.hashCellSize;
    float radius = params.minGatherRadius;

    // Weighted spectral accumulation + tangential dipole for edge detection
    float specOut[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float3 dipoleT[10];   // tangential dipole (projected to surface plane)
    float  sumEnergy[10];
    for (int b = 0; b < 10; b++) {
        dipoleT[b] = float3(0, 0, 0);
        sumEnergy[b] = 0.0;
    }
    int totalCount = 0;
    float finalRadius = params.maxGatherRadius;

    int3 oldMin = int3(1, 1, 1);
    int3 oldMax = int3(-1, -1, -1);

    int iterCap = 0;
    while (iterCap < 32) {
        iterCap++;
        int3 newMin = computeCell(centerPos - float3(radius), cellSz);
        int3 newMax = computeCell(centerPos + float3(radius), cellSz);

        float r2 = radius * radius;
        float invR = 1.0 / max(radius, 0.0001);

        for (int cx = newMin.x; cx <= newMax.x; cx++) {
            bool inOldX = (cx >= oldMin.x && cx <= oldMax.x);
            for (int cy = newMin.y; cy <= newMax.y; cy++) {
                bool inOldY = (cy >= oldMin.y && cy <= oldMax.y);
                for (int cz = newMin.z; cz <= newMax.z; cz++) {
                    bool inOldZ = (cz >= oldMin.z && cz <= oldMax.z);
                    if (inOldX && inOldY && inOldZ) continue;

                    uint nHash = computeCellHash(int3(cx, cy, cz));
                    uint nBase = nHash * FLOATS_PER_CELL;
                    float cellCnt = cellPhotonData[nBase + 22];
                    if (cellCnt == 0.0) continue;

                    float3 phPos = float3(cellPhotonData[nBase + 0],
                                          cellPhotonData[nBase + 1],
                                          cellPhotonData[nBase + 2]);
                    float3 d = phPos - centerPos;
                    float d2 = dot(d, d);
                    if (d2 >= r2) continue;

                    float3 phNrm = float3(cellPhotonData[nBase + 4],
                                          cellPhotonData[nBase + 5],
                                          cellPhotonData[nBase + 6]);
                    if (dot(phNrm, centerNrm) < 0.2) continue;

                    float w = max(0.0, 1.0 - sqrt(d2) * invR);

                    // Tangential projection: remove normal component of d.
                    // Photons lie on 2D surfaces — 3D offset caused by surface
                    // curvature is NOT an illumination edge.
                    float3 dTan = d - dot(d, centerNrm) * centerNrm;

                    for (int b = 0; b < 10; b++) {
                        float sb = cellPhotonData[nBase + 12 + b];
                        specOut[b]   += sb * w;
                        sumEnergy[b] += sb;
                        dipoleT[b]   += dTan * sb;
                    }

                    totalCount += int(cellCnt);
                }
            }
        }

        if (totalCount >= params.maxGatherPhotons) { finalRadius = radius; break; }
        if (radius >= params.maxGatherRadius)         { finalRadius = radius; break; }

        // Edge detection: only after minGatherPhotons have been accumulated,
        // suppressing false triggers from statistical noise at small radii.
        if (totalCount >= params.minGatherPhotons) {
            float maxE = 0.0;
            for (int b = 0; b < 10; b++) maxE = max(maxE, sumEnergy[b]);

            if (maxE > 1e-6) {
                bool edge = false;
                float scoInvR = 1.0 / max(radius, 0.0001);

                for (int b = 0; b < 10; b++) {
                    if (sumEnergy[b] < 1e-6) continue;
                    float3 Dk = dipoleT[b] / sumEnergy[b];
                    float score = length(Dk) * scoInvR;
                    float Nk = float(totalCount) * (sumEnergy[b] / maxE);
                    float noiseThresh = params.confidence
                                      / sqrt(max(1.0, Nk));
                    if (score > max(params.minEdgeScore, noiseThresh)) {
                        edge = true;
                        break;
                    }
                }

                if (edge) { finalRadius = radius; break; }
            }
        }

        oldMin = newMin;
        oldMax = newMax;
        float prevR = radius;
        radius = min(radius + cellSz, params.maxGatherRadius);
        if (radius <= prevR) { finalRadius = radius; break; }
    }

    // Area normalization: photon map radiance estimate.
    // Area normalization: divide by actually-sampled kernel integral.
    // Less sampled area (edge-stopped) → less energy, Decay ∝ r².
    // Linear kernel w(t)=1-t/r has effective area π·r²/3.
    float area = PI * finalRadius * finalRadius / 3.0;
    float invArea = 1.0 / max(area, 1e-10);

    for (int b = 0; b < NUM_SPECTRAL_BUCKETS; b++)
        gatheredCellData[outBase + b] = specOut[b] * invArea;
}
