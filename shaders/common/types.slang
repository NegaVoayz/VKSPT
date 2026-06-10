// Spectral constants & shared types

const int LAMBDA_MIN           = 380;
const int LAMBDA_MAX           = 780;
const int SPECTRAL_STACK_MAX    = 256;
const int NUM_SPECTRAL_BUCKETS = 10;

const int MATERIAL_DIELECTRIC   = 0;
const int MATERIAL_METAL        = 1;
const int MATERIAL_LAMBERTIAN   = 2;
const int MATERIAL_CHECKERBOARD = 3;

struct GpuMaterial {
    vec4 cauchyA, cauchyB;
    vec4 absorpA, absorpB;
    vec4 albedo, params;
};

struct RayState {
    vec3 origin;
    vec3 direction;
    vec3 last_split;
    vec3 dispersion;
    vec3 last_geom;
    vec3 geomDisp;
    int  lam_start;
    int  lam_end;
    float throughput;
    float decay_c1;
    float decay_c2;
    int  bounce;
    int  generation;
};

struct GpuLight {
    vec4 pos_type;
    vec4 color_intensity;
    vec4 dir_inner;
    vec4 outer_range;
};

const int LIGHT_POINT       = 0;
const int LIGHT_DIRECTIONAL = 1;
const int LIGHT_SPOT        = 2;
const int LIGHT_AMBIENT     = 3;
