// Dielectric material handler — refraction, Fresnel, adaptive spectral splitting
import common.types;
import math.spectral_math;
import math.absorption;
import math.legendre_decay;
import geometry.dispersion;

[ForceInline]
public void handleDielectric(
    inout BounceState bs,
    inout int stackTop,
    inout RayState rayStack[SPECTRAL_STACK_MAX],
    float3 pos, float3 rawNrm, GpuMaterial mat,
    float tHit, int bounce, int2 sz,
    float fovTan, float splitMult, float forceSplitWidth)
{
    bool entering = dot(rawNrm, bs.rayD) < 0.0;
    float3 nrm = entering ? rawNrm : -rawNrm;

    int lamMid = (bs.lamStart + bs.lamEnd) / 2;
    float ior_start = cauchyIOR(mat, float(bs.lamStart));
    float ior_end   = cauchyIOR(mat, float(bs.lamEnd));
    float ior_mid   = cauchyIOR(mat, float(lamMid));
    float eta_mid   = entering ? (1.0 / ior_mid) : ior_mid;
    float deta      = (entering ? (1.0/ior_end - 1.0/ior_start) : (ior_end - ior_start));

    float lamMidF  = float(bs.lamStart + bs.lamEnd) * 0.5;
    float halfW    = float(bs.lamEnd - bs.lamStart) * 0.5;
    bool decayPoorFit = applyVolumeDecay(bs.throughput, bs.decayC1, bs.decayC2,
                                          lamMidF, halfW,
                                          mat.absorpA.y, mat.absorpB.y, tHit);

    float3 newDir;
    if (refractDir(bs.rayD, nrm, eta_mid, newDir)) {
        float3 newDisp = computeDispersion(bs.rayD, nrm, eta_mid, deta);
        float3 newGeom = computeGeomDispersion(bs.rayD, nrm, eta_mid);

        bs.lastSplit = rotateToNewFrame(bs.lastSplit, bs.rayD, newDir);
        bs.disp      = rotateToNewFrame(bs.disp, bs.rayD, newDir) + newDisp;
        bs.lastGeom  = rotateToNewFrame(bs.lastGeom, bs.rayD, newDir);
        bs.geomD     = rotateToNewFrame(bs.geomD, bs.rayD, newDir) + newGeom;

        float cosTheta = abs(dot(bs.rayD, nrm));
        float R0 = ((ior_mid - 1.0) / (ior_mid + 1.0));
        R0 = R0 * R0;
        float fresnelR = R0 + (1.0 - R0) * pow(1.0 - cosTheta, 5.0);

        float reflThroughput = bs.throughput + log(fresnelR);
        if (stackTop < SPECTRAL_STACK_MAX && reflThroughput > -5.298317) {
            float3 reflDir = reflectDir(bs.rayD, nrm);
            RayState refl;
            refl.origin      = pos + nrm * 0.001 + reflDir * 0.01;
            refl.direction   = reflDir;
            refl.last_split  = bs.lastSplit - 2.0 * dot(bs.lastSplit, nrm) * nrm;
            refl.dispersion  = bs.disp - 2.0 * dot(bs.disp, nrm) * nrm;
            refl.last_geom   = bs.lastGeom - 2.0 * dot(bs.lastGeom, nrm) * nrm;
            refl.geomDisp    = bs.geomD - 2.0 * dot(bs.geomD, nrm) * nrm;
            refl.lam_start   = bs.lamStart;
            refl.lam_end     = bs.lamEnd;
            refl.throughput  = reflThroughput;
            refl.decay_c1    = bs.decayC1;
            refl.decay_c2    = bs.decayC2;
            refl.bounce      = bounce + 1;
            refl.generation  = bs.generation;
            rayStack[stackTop++] = refl;
        }
        bs.throughput += log(max(1.0 - fresnelR, 1e-10));
        // c1,c2 unchanged — Fresnel is wavelength-independent

        // ---- Split decision: cheap conditions first, short-circuit ----
        int  lamRange  = bs.lamEnd - bs.lamStart;
        bool forceWide = (int(forceSplitWidth) > 0) && (lamRange > int(forceSplitWidth));

        // Predicted dispersion (cheap MADs, needed in split body)
        float estDist     = max(tHit, 50.0);
        float3 predictedGeom = bs.lastGeom + bs.geomD * estDist;
        float3 predictedSep  = bs.lastSplit + bs.disp * estDist;

        bool shouldSplit = forceWide;
        float geomLen = 0.0, specLen = 0.0;
        if (!shouldSplit) {
            // Only compute lengths (sqrt) when not already splitting
            geomLen            = length(predictedGeom);
            specLen            = length(predictedSep);
            float pixHalfWidth = fovTan / float(sz.x);
            float pixThresh    = estDist * pixHalfWidth * 0.5 * splitMult;
            shouldSplit        = max(geomLen, specLen) > pixThresh;
        }
        shouldSplit = shouldSplit && lamRange >= 10
                      && !decayIsDead(bs.throughput, bs.decayC2)
                      && bs.generation < 8;

        if (shouldSplit) {
            bool geomDriven     = (geomLen > specLen);
            bool spectralDriven = (!forceWide);  // forceWide ⇒ spectral split

            int midWl = (bs.lamStart + bs.lamEnd) / 2;
            int nextGen = bs.generation + 1;

            if (geomDriven && !spectralDriven) {
                nextGen = bs.generation + 4;
                RayState gchild;
                gchild.origin      = pos + nrm * 0.001 + newDir * 0.01;
                gchild.direction   = newDir;
                gchild.last_split  = predictedSep;
                gchild.dispersion  = bs.disp;
                gchild.last_geom   = predictedGeom;
                gchild.geomDisp    = bs.geomD;
                gchild.lam_start   = bs.lamStart;
                gchild.lam_end     = bs.lamEnd;
                gchild.throughput  = bs.throughput - 0.693147; // ln(0.5)
                gchild.decay_c1    = bs.decayC1;
                gchild.decay_c2    = bs.decayC2;
                gchild.bounce      = bounce + 1;
                gchild.generation  = nextGen;
                if (stackTop < SPECTRAL_STACK_MAX) {
                    rayStack[stackTop++] = gchild;
                    bs.throughput -= 0.693147;
                    bs.generation = nextGen;
                }
            } else {
                RayState violet;
                violet.origin      = pos + nrm * 0.001 + newDir * 0.01;
                violet.direction   = newDir;
                violet.last_split  = predictedSep;
                violet.dispersion  = bs.disp;
                violet.last_geom   = predictedGeom;
                violet.geomDisp    = bs.geomD;
                violet.lam_start   = bs.lamStart;
                violet.lam_end     = midWl;
                reprojectDecayLeft(bs.throughput, bs.decayC1, bs.decayC2,
                                   violet.throughput, violet.decay_c1, violet.decay_c2);
                violet.bounce      = bounce + 1;
                violet.generation  = nextGen;

                if (stackTop < SPECTRAL_STACK_MAX) {
                    rayStack[stackTop++] = violet;
                    reprojectDecayRight(bs.throughput, bs.decayC1, bs.decayC2,
                                        bs.throughput, bs.decayC1, bs.decayC2);
                    bs.lamStart   = midWl;
                    bs.generation = nextGen;
                }
            }
        }

        bs.rayD    = newDir;
        bs.rayO    = pos + nrm * 0.001 + newDir * 0.01;
    }
    else {
        // TIR — total internal reflection
        newDir     = reflectDir(bs.rayD, nrm);
        bs.lastSplit  = bs.lastSplit - 2.0 * dot(bs.lastSplit, nrm) * nrm;
        bs.disp       = bs.disp - 2.0 * dot(bs.disp, nrm) * nrm;
        bs.lastGeom   = bs.lastGeom - 2.0 * dot(bs.lastGeom, nrm) * nrm;
        bs.geomD      = bs.geomD - 2.0 * dot(bs.geomD, nrm) * nrm;
        bs.rayD       = newDir;
        bs.rayO       = pos + nrm * 0.001 + newDir * 0.01;
    }
}
