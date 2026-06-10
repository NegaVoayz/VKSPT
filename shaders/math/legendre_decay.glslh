// Legendre decay expansion: 2nd-order Legendre series D(λ)=exp(-τ(λ))

bool applyVolumeDecay(inout float c0, inout float c1, inout float c2,
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

void reprojectDecay(float parentC0, float parentC1, float parentC2,
                    float parentMid, float parentHalf,
                    float childMid,  float childHalf,
                    out float childC0, out float childC1, out float childC2) {
    const float SQRT_3_5 = 0.77459667;
    const float W0 = 5.0 / 9.0, W1 = 8.0 / 9.0, W2 = 5.0 / 9.0;
    float Dvals[3];
    for (int i = 0; i < 3; i++) {
        float xk = (i == 0) ? -SQRT_3_5 : ((i == 1) ? 0.0 : SQRT_3_5);
        float lam = childMid + childHalf * xk;
        float xParent = (lam - parentMid) / max(parentHalf, 0.001);
        xParent = clamp(xParent, -1.0, 1.0);
        Dvals[i] = evalDecay(parentC0, parentC1, parentC2, xParent);
    }
    childC0 = 0.5 * (W0 * Dvals[0] + W1 * Dvals[1] + W2 * Dvals[2]);
    childC1 = 1.5 * (W0 * Dvals[0] * (-SQRT_3_5) + W2 * Dvals[2] * SQRT_3_5);
    childC2 = 2.5 * (W0 * Dvals[0] * 0.4 + W1 * Dvals[1] * (-0.5) + W2 * Dvals[2] * 0.4);
}

void scaleDecay(inout float c0, inout float c1, inout float c2, float s) {
    c0 *= s; c1 *= s; c2 *= s;
}
