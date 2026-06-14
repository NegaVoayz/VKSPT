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

// Pass type for shared RT pipeline
public static const int PASS_CAMERA = 0;
public static const int PASS_PHOTON = 1;

// Photon map
public static const int MAX_PHOTONS = 2097152;  // 2M cap
public static const int HASH_TABLE_SIZE = 262144;  // 2^18, spatial hash buckets

public struct PhotonRecord {
    public float4 phPos;            // xyz = hit position, w = unused
    public float4 phNrm;            // xyz = surface normal, w = unused
    public float4 phDir;            // xyz = photon arrival direction, w = unused
    public float4 spectral0_3;      // fp32 spectral buckets 0-3
    public float4 spectral4_7;      // fp32 spectral buckets 4-7
    public float2 spectral8_9;      // fp32 spectral buckets 8-9
};
// Total: 88 bytes
