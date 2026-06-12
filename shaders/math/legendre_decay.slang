// Legendre log-decay expansion
// Coefficients (c0,c1,c2) now represent L(x) = ln(transmission(x)),
// a 2nd-order Legendre series on x ∈ [-1,1].
// All operations on the log-domain: mul→add, div→subtract.

import math.spectral_math;

// Apply volume absorption: L_new = L_old − α(λ)·t
// Fits α(λ) = absorpA + absorpB/λ² into Legendre on the ray's wavelength
// range via 3-point Gauss-Legendre quadrature, then subtracts coefficients.
// Returns false (absorption-driven split detection deferred for log space).
public bool applyVolumeDecay(inout float c0, inout float c1, inout float c2,
                       float lamMid, float halfWidth,
                       float absorpA, float absorpB, float tHit) {
    if (absorpA <= 0.0 && absorpB <= 0.0) return false;
    if (halfWidth < 1.0) return false;

    const float SQRT_3_5 = 0.77459667;
    const float W0 = 5.0 / 9.0, W1 = 8.0 / 9.0, W2 = 5.0 / 9.0;

    // Fit -α(λ)·t into Legendre on this wavelength range
    float aVals[3];
    for (int i = 0; i < 3; i++) {
        float xk = (i == 0) ? -SQRT_3_5 : ((i == 1) ? 0.0 : SQRT_3_5);
        float lam = lamMid + halfWidth * xk;
        float lu  = lam * 1e-3;
        aVals[i] = -(absorpA + absorpB / (lu * lu)) * tHit;
    }

    float a0 = 0.5 * (W0 * aVals[0] + W1 * aVals[1] + W2 * aVals[2]);
    float a1 = 1.5 * (W0 * aVals[0] * (-SQRT_3_5) + W2 * aVals[2] * SQRT_3_5);
    float p2_0 = 0.4, p2_1 = -0.5, p2_2 = 0.4;
    float a2 = 2.5 * (W0 * aVals[0] * p2_0 + W1 * aVals[1] * p2_1 + W2 * aVals[2] * p2_2);

    c0 += a0; c1 += a1; c2 += a2;
    return false; // absorption-driven split deferred in log space
}

// Binary split at midpoint: child = LEFT half of parent [lamStart, midWl]
//   a = -0.5, b = 0.5
[ForceInline]
public void reprojectDecayLeft(float parentC0, float parentC1, float parentC2,
                                out float childC0, out float childC1, out float childC2) {
    childC0 = parentC0 - 0.5 * parentC1;
    childC1 = 0.5 * parentC1 - 0.75 * parentC2;
    childC2 = 0.25 * parentC2;
}

// Binary split at midpoint: child = RIGHT half of parent [midWl, lamEnd]
//   a = 0.5, b = 0.5
[ForceInline]
public void reprojectDecayRight(float parentC0, float parentC1, float parentC2,
                                 out float childC0, out float childC1, out float childC2) {
    childC0 = parentC0 + 0.5 * parentC1;
    childC1 = 0.5 * parentC1 + 0.75 * parentC2;
    childC2 = 0.25 * parentC2;
}

// General affine reprojection for non-binary splits.
// Prefer reprojectDecayLeft/Right for the common binary-split case.
public void reprojectDecay(float parentC0, float parentC1, float parentC2,
                    float parentMid, float parentHalf,
                    float childMid,  float childHalf,
                    out float childC0, out float childC1, out float childC2) {
    float invPH = 1.0 / max(parentHalf, 0.001);
    float a = (childMid - parentMid) * invPH;
    float b = childHalf * invPH;

    float a2 = a * a;
    float b2 = b * b;
    float ab = a * b;

    childC2 = parentC2 * b2;
    childC1 = parentC1 * b + 3.0 * parentC2 * ab;
    childC0 = parentC0 + parentC1 * a + parentC2 * (1.5 * a2 - 0.5 + 0.5 * b2);
}

// Log-space: multiply decay by scalar s → add ln(s) to c0 only
[ForceInline]
public void scaleDecay(inout float c0, float s) {
    c0 += log(max(s, 1e-10));
}
