// Legendre decay expansion
import math.spectral_math;
// 2nd-order Legendre series D(λ)=exp(-τ(λ))

public bool applyVolumeDecay(inout float c0, inout float c1, inout float c2,
                       float lamMid, float halfWidth,
                       float absorpA, float absorpB, float tHit) {
    float oldC0 = c0, oldC1 = c1, oldC2 = c2;
    const float SQRT_3_5 = 0.77459667;
    const float W0 = 5.0 / 9.0, W1 = 8.0 / 9.0, W2 = 5.0 / 9.0;
    float newD[3];
    for (int i = 0; i < 3; i++) {
        float xk = (i == 0) ? -SQRT_3_5 : ((i == 1) ? 0.0 : SQRT_3_5);
        float lam = lamMid + halfWidth * xk;
        float lu  = lam * 1e-3;
        float alpha = absorpA + absorpB / (lu * lu);
        float D_old = evalDecay(c0, c1, c2, xk);
        newD[i] = D_old * exp(-alpha * tHit);
    }
    c0 = 0.5 * (W0 * newD[0] + W1 * newD[1] + W2 * newD[2]);
    c1 = 1.5 * (W0 * newD[0] * (-SQRT_3_5) + W2 * newD[2] * SQRT_3_5);
    float p2_0 = 0.4, p2_1 = -0.5, p2_2 = 0.4;
    c2 = 2.5 * (W0 * newD[0] * p2_0 + W1 * newD[1] * p2_1 + W2 * newD[2] * p2_2);
    if (halfWidth < 1.0) return false;
    float lamCp  = lamMid + halfWidth * 0.5;
    float luCp   = lamCp * 1e-3;
    float alphaCp = absorpA + absorpB / (luCp * luCp);
    float oldD_cp = evalDecay(oldC0, oldC1, oldC2, 0.5);
    float D_exact = oldD_cp * exp(-alphaCp * tHit);
    float D_fit   = evalDecay(c0, c1, c2, 0.5);
    float err     = abs(D_exact - D_fit) / max(c0, 0.001);
    return err > 0.02;
}

// Binary split at midpoint: child = LEFT half of parent [lamStart, midWl]
// All splits use the exact midpoint, so a,b are constants:
//   a = -0.5, b = 0.5  →  c0'=c0-0.5c1, c1'=0.5c1-0.75c2, c2'=0.25c2
[ForceInline]
public void reprojectDecayLeft(float parentC0, float parentC1, float parentC2,
                                out float childC0, out float childC1, out float childC2) {
    childC0 = parentC0 - 0.5 * parentC1;
    childC1 = 0.5 * parentC1 - 0.75 * parentC2;
    childC2 = 0.25 * parentC2;
}

// Binary split at midpoint: child = RIGHT half of parent [midWl, lamEnd]
//   a = 0.5, b = 0.5  →  c0'=c0+0.5c1, c1'=0.5c1+0.75c2, c2'=0.25c2
[ForceInline]
public void reprojectDecayRight(float parentC0, float parentC1, float parentC2,
                                 out float childC0, out float childC1, out float childC2) {
    childC0 = parentC0 + 0.5 * parentC1;
    childC1 = 0.5 * parentC1 + 0.75 * parentC2;
    childC2 = 0.25 * parentC2;
}

// General affine reprojection: parent coord x ∈ [-1,1] → child coord z ∈ [-1,1]
// via x = a + b*z, for non-binary splits (kept for future use).
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

public void scaleDecay(inout float c0, inout float c1, inout float c2, float s) {
    c0 *= s; c1 *= s; c2 *= s;
}
