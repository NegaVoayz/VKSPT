#pragma once

#include <cstdint>

// =============================================================================
// CPU-side spectral tracing data structures.
//
// These mirror the design doc (项目设计文档) definitions and the GPU-side
// structs in spectral_common.slang.
//
// Phase 2 uses Material for the material UBO.
// SpectralRay and RayAction are reference definitions for Phase 3+.
// =============================================================================

/// Per-wavelength material properties uploaded to a GPU uniform buffer.
/// Must match GpuMaterial in spectral_common.slang (std140, 64 bytes).
struct alignas(16) GpuMaterialCPU {
    float cauchyA[4];    // A coefficients (RGB), w unused
    float cauchyB[4];    // B coefficients (RGB), w unused — B for λ in μm
    float albedo[4];     // base colour / reflectance
    float params[4];     // x=ior, y=roughness, z=type, w unused
};

/// Material type enum (matches shader constants).
enum class MaterialType : int32_t {
    DIELECTRIC  = 0,
    METAL       = 1,
    LAMBERTIAN  = 2,
};

/// Full material definition (host-side).
/// For Phase 2, converted to GpuMaterialCPU for the UBO.
struct Material {
    MaterialType type = MaterialType::DIELECTRIC;

    // Cauchy dispersion coefficients: n(λ) = A + B/λ²
    // Per-channel (RGB), B is given for λ in μm
    float cauchyA[3] = {1.517f, 1.517f, 1.517f};  // BK7 glass default
    float cauchyB[3] = {0.0045f, 0.0045f, 0.0045f};

    // Base IOR (used for non-dielectric or as fallback)
    float ior = 1.517f;

    // Optical properties
    float albedo[3]    = {0.7f, 0.5f, 0.3f};
    float roughness    = 0.0f;
    float absorption[3] = {0.0f, 0.0f, 0.0f};
};

/// Spectral ray — definition from design doc §3.1.
/// Phase 2 uses this as reference; the shader currently traces individual
/// wavelength samples rather than full spectral rays. Phase 3 will use this
/// for the CPU-side ray queue.
struct SpectralRay {
    // --- geometric ---
    float origin[3];
    float direction[3];

    // --- dispersion ---
    float dispersion[3];    // lateral chromatic spread per unit distance

    // --- spectral ---
    int32_t lambda_start;   // nm, inclusive [380, 780]
    int32_t lambda_end;     // nm, inclusive

    // --- energy ---
    float energy[3];        // RGB energy weight (attenuated by absorption)

    // --- accumulated lateral separation ---
    float last_split[3];    // accumulated since last split event

    // --- control ---
    int32_t bounce;
    int32_t generation;     // split generation counter
};

/// Action classification for a spectral ray after a surface interaction.
/// Matches design doc §3.3.
enum class RayAction : int32_t {
    TERMINATE = 0,   // energy exhausted or max bounces reached
    SPLIT     = 1,   // chromatic separation exceeds pixel threshold
    REFRACT   = 2,   // transmit through dielectric
    REFLECT   = 3,   // reflect from surface
    MISS      = 4,   // no intersection — accumulate environment
};
