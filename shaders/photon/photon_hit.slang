// Photon-mode hit handling: store at diffuse surfaces, bounce at specular.
import common.types;
import common.bindings;
import math.sampling;
import math.spectral_math;
import math.color;
import rt.payload;
import rt.material_handlers;

void storePhoton(inout Payload p, float3 pos, float3 fNrm)
{
    uint slot;
    InterlockedAdd(photonCounter[0], 1, slot);
    if (slot >= MAX_PHOTONS) return;

    PhotonRecord rec;
    rec.phPos = float4(pos, 0.0);
    rec.phNrm = float4(fNrm, 0.0);
    rec.phDir = float4(normalize(WorldRayDirection()), 0.0);

    // Spectral energy: Wiener spectrum × Legendre attenuation.
    float lamMid = float(p.lamStart + p.lamEnd) * 0.5;
    float halfW  = float(p.lamEnd - p.lamStart) * 0.5;
    float invHW  = 1.0 / max(halfW, 0.001);

    float specBucket[10];
    for (int b = 0; b < 10; b++) {
        float bS = float(LAMBDA_MIN + 40 * b);
        float bE = bS + 40.0;
        float oS = max(float(p.lamStart), bS);
        float oE = min(float(p.lamEnd), bE);
        if (oS >= oE) { specBucket[b] = 0.0; continue; }

        float lamMidB = (oS + oE) * 0.5;
        float x = clamp((lamMidB - lamMid) * invHW, -1.0, 1.0);
        float S = estimateSpectrum(lamMidB, p.lightRGB);
        float D = evalDecay(p.throughput, p.decayC1, p.decayC2, x);
        specBucket[b] = S * D * (oE - oS);
    }

    rec.spectral0_3 = float4(specBucket[0], specBucket[1], specBucket[2], specBucket[3]);
    rec.spectral4_7 = float4(specBucket[4], specBucket[5], specBucket[6], specBucket[7]);
    rec.spectral8_9 = float2(specBucket[8], specBucket[9]);
    photonBuffer[slot] = rec;
}

void handlePhotonHit(inout Payload p, float3 pos, float3 fNrm, float3 nrm, int matType)
{
    if (p.bounce >= params.photonMaxBounces || decayIsDead(p.throughput, p.decayC2))
        return;

    if (p.bounce >= 3) {
        float linTp = exp(p.throughput - 0.5 * p.decayC2);
        float survive = max(linTp, 0.1);
        if (rand1(p.rng) > survive) return;
        p.throughput -= log(survive);
    }

    // p.D already propagated (D + V*hitT) by apply_diff in ClosestHitMain.

    GpuMaterial mat = materials.items[clamp(uint(matType), 0u,
        max(0u, uint(params.materialCount) - 1u))];

    if (matType == MATERIAL_DIELECTRIC) {
        p.specularChain++;
        handleDielectric(p, pos, nrm, mat);
    }
    else if (matType == MATERIAL_METAL) {
        p.specularChain++;
        handleMetal(p, pos, fNrm, mat, p.rng);
    }
    else if (matType == MATERIAL_LAMBERTIAN || matType == MATERIAL_CHECKERBOARD) {
        storePhoton(p, pos, fNrm);
    }
}
