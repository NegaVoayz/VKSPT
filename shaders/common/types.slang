// Spectral constants & shared types

public static const int LAMBDA_MIN           = 380;
public static const int LAMBDA_MAX           = 780;
public static const int SPECTRAL_STACK_MAX    = 256;
public static const int NUM_SPECTRAL_BUCKETS = 10;

public static const int MATERIAL_DIELECTRIC   = 0;
public static const int MATERIAL_METAL        = 1;
public static const int MATERIAL_LAMBERTIAN   = 2;
public static const int MATERIAL_CHECKERBOARD = 3;

public struct GpuMaterial {
    public float4 cauchyA, cauchyB;
    public float4 absorpA, absorpB;
    public float4 albedo, params;
    public float4 precalc;       // .x = ln(0.07+0.93*lum) for lambertian
};

public struct GpuLight {
    public float4 posType;
    public float4 colorIntensity;
    public float4 dirInner;
    public float4 outerRange;
};

public static const int LIGHT_POINT       = 0;
public static const int LIGHT_DIRECTIONAL = 1;
public static const int LIGHT_SPOT        = 2;
public static const int LIGHT_AMBIENT     = 3;
