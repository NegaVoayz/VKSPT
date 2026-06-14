// Hash grid dispatch 4: aggregate all photons per cell into one representative.
// Uses raw RWStructuredBuffer<float> (binding 18) to avoid PhotonRecord half-type
// crash.  Layout per cell (23 floats): pos(4) + nrm(4) + dir(4) + spec(10) + cnt(1).
// Count at base+22.
import common.types;
import common.bindings;

static const uint FLOATS_PER_CELL = 23;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint hash = tid.x;
    if (hash >= HASH_TABLE_SIZE) return;

    uint base = hash * FLOATS_PER_CELL;
    uint cnt = hashCellData[hash * 2 + 0];
    uint off = hashCellData[hash * 2 + 1];

    if (cnt == 0) {
        for (uint k = 0; k < FLOATS_PER_CELL; k++) {
            cellPhotonData[base + k] = 0.0;
        }
        return;
    }

    float3 posSum = float3(0, 0, 0);
    float3 nrmSum = float3(0, 0, 0);
    float3 dirSum = float3(0, 0, 0);
    float  specSum[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (uint j = 0; j < cnt; j++) {
        uint idx = off + j;
        if (idx >= MAX_PHOTONS) break;
        PhotonRecord ph = photonBuffer[sortedPhotonIndices[idx]];

        posSum += ph.phPos.xyz;
        nrmSum += ph.phNrm.xyz;
        dirSum += ph.phDir.xyz;

        specSum[0] += ph.spectral0_3.x;
        specSum[1] += ph.spectral0_3.y;
        specSum[2] += ph.spectral0_3.z;
        specSum[3] += ph.spectral0_3.w;
        specSum[4] += ph.spectral4_7.x;
        specSum[5] += ph.spectral4_7.y;
        specSum[6] += ph.spectral4_7.z;
        specSum[7] += ph.spectral4_7.w;
        specSum[8] += ph.spectral8_9.x;
        specSum[9] += ph.spectral8_9.y;
    }

    float invN = 1.0 / float(cnt);

    cellPhotonData[base + 0] = posSum.x * invN;
    cellPhotonData[base + 1] = posSum.y * invN;
    cellPhotonData[base + 2] = posSum.z * invN;
    cellPhotonData[base + 3] = 0.0;  // w unused

    float nrmLen = length(nrmSum);
    float3 n = (nrmLen > 1e-10) ? (nrmSum / nrmLen) : float3(0, 1, 0);
    cellPhotonData[base + 4] = n.x;
    cellPhotonData[base + 5] = n.y;
    cellPhotonData[base + 6] = n.z;
    cellPhotonData[base + 7] = 0.0;

    float dirLen = length(dirSum);
    float3 d = (dirLen > 1e-10) ? (dirSum / dirLen) : float3(0, 0, 1);
    cellPhotonData[base + 8]  = d.x;
    cellPhotonData[base + 9]  = d.y;
    cellPhotonData[base + 10] = d.z;
    cellPhotonData[base + 11] = 0.0;

    for (int b = 0; b < 10; b++)
        cellPhotonData[base + 12 + b] = specSum[b];

    cellPhotonData[base + 22] = float(cnt);  // original photon count
}
