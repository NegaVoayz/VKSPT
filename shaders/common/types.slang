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
};

public struct RayState {
    public float3 origin;
    public float3 direction;
    public float3 last_split;
    public float3 dispersion;
    public float3 last_geom;
    public float3 geomDisp;
    public int  lam_start;
    public int  lam_end;
    public float throughput;
    public float decay_c1;
    public float decay_c2;
    public int  bounce;
    public int  generation;
};

// Mutable per-bounce state passed to material handlers
public struct BounceState {
    public float3 rayO, rayD;
    public float  throughput;
    public float  decayC1, decayC2;
    public float3 lastSplit, disp;
    public float3 lastGeom, geomD;
    public int    lamStart, lamEnd;
    public int    generation;
    public float3 colorAccum;
};

[ForceInline]
public BounceState packBs(
    float3 rayO, float3 rayD,
    float throughput, float decayC1, float decayC2,
    float3 lastSplit, float3 disp,
    float3 lastGeom, float3 geomD,
    int lamStart, int lamEnd, int generation, float3 colorAccum)
{
    BounceState bs;
    bs.rayO = rayO; bs.rayD = rayD;
    bs.throughput = throughput; bs.decayC1 = decayC1; bs.decayC2 = decayC2;
    bs.lastSplit = lastSplit; bs.disp = disp;
    bs.lastGeom = lastGeom; bs.geomD = geomD;
    bs.lamStart = lamStart; bs.lamEnd = lamEnd;
    bs.generation = generation; bs.colorAccum = colorAccum;
    return bs;
}

[ForceInline]
public void unpackBs(
    inout float3 rayO, inout float3 rayD,
    inout float throughput, inout float decayC1, inout float decayC2,
    inout float3 lastSplit, inout float3 disp,
    inout float3 lastGeom, inout float3 geomD,
    inout int lamStart, inout int lamEnd,
    inout int generation, inout float3 colorAccum,
    BounceState bs)
{
    rayO = bs.rayO; rayD = bs.rayD;
    throughput = bs.throughput; decayC1 = bs.decayC1; decayC2 = bs.decayC2;
    lastSplit = bs.lastSplit; disp = bs.disp;
    lastGeom = bs.lastGeom; geomD = bs.geomD;
    lamStart = bs.lamStart; lamEnd = bs.lamEnd;
    generation = bs.generation; colorAccum = bs.colorAccum;
}

public struct GpuLight {
    public float4 pos_type;
    public float4 color_intensity;
    public float4 dir_inner;
    public float4 outer_range;
};

public static const int LIGHT_POINT       = 0;
public static const int LIGHT_DIRECTIONAL = 1;
public static const int LIGHT_SPOT        = 2;
public static const int LIGHT_AMBIENT     = 3;
