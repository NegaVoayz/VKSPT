# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**自适应光谱追踪渲染器** (Adaptive Spectral Tracing Renderer) — a physically correct dispersion renderer using Vulkan 1.4 Ray Query, targeting NVIDIA RTX 4060+ (Ada Lovelace, 3rd-gen RT Core).

Core idea: trace spectral rays (380nm–780nm) through dispersive media, adaptively splitting rays when chromatic separation exceeds a pixel threshold at the next surface interaction.

## Build & Toolchain

- **C++ standard**: C++20
- **Build system**: CMake 4.3.1 (no `CMakeLists.txt` exists yet at repo root — project is in design phase)
- **Vulkan SDK**: `C:\VulkanSDK\1.4.341.1`
- **GPU target**: NVIDIA RTX 4060+

### Dependencies (from Vulkan SDK)

- **Vulkan 1.4** + Vulkan-Hpp (vulkan.hpp) — RAII C++ bindings
- **SDL3** — windowing (`VulkanSDK\1.4.341.1\Include\SDL3`)
- **GLM** — math (`VulkanSDK\1.4.341.1\Include\glm`)
- **Slang** — shader compiler (`VulkanSDK\1.4.341.1\Include\slang`)
- **SPIR-V** — shader binary format

### Third-party (vendored in `third-party/`)

| Library | Path | Purpose |
|---|---|---|
| tinyobjloader 1.0.6 | `third-party/tinyobjloader-1.0.6/` | OBJ mesh loading |
| tinyxml2 11.0.0 | `third-party/tinyxml2-11.0.0/` | XML scene config parsing |
| stb | `third-party/stb/` | stb_image for texture loading |

### Assets (`assets/`)

- **OBJ meshes**: duck, bunny, dragon, venus, asschercut, fudanlogo
- **Scene configs**: `SceneConfig*.xml` — scene description files parsed via tinyxml2 (camera, lights, object transforms, materials, render settings)

## Architecture

The renderer follows a 4-stage GPU pipeline per bounce iteration, all within compute shaders using `VK_KHR_ray_query`:

1. **Ray Sorting** — classify active rays by `RayAction` (SPLIT/REFRACT/REFLECT/MISS/TERMINATE) into buckets via atomic counters + prefix sum. Each bucket dispatches separately to reduce warp divergence.
2. **Ray Query Intersection** — `traceRayEXT` against BLAS/TLAS, return hit point, normal, material ID.
3. **Hit Handling** — dispatch by `RayAction`: split rays at wavelength midpoint (energy split proportional to bandwidth), compute refraction direction and dispersion vector, reflect, or accumulate environment light on miss.
4. **Ray Propagation** — advance origin by direction × distance, accumulate lateral chromatic separation (`last_split += dispersion * distance`).

### Key Data Structures (see design doc for full definitions)

- **`SpectralRay`** — origin, direction, dispersion vector, wavelength range [λ_start, λ_end] (integer nm), RGB energy, `last_split` (lateral separation), bounce count, split generation.
- **`Material`** — type enum (DIELECTRIC/METAL/LAMBERTIAN), Cauchy coefficients (n = A + B/λ²) per RGB channel, albedo, absorption, roughness, base IOR.
- **`RayAction`** — enum: TERMINATE, SPLIT, REFRACT, REFLECT, MISS.

### Split Decision Algorithm

A ray splits when ALL of:
1. Predicted lateral separation at the next surface event exceeds pixel-size threshold
2. Wavelength range ≥ 10 nm (prevents infinite splitting)
3. Energy magnitude ≥ 0.05 (low-energy rays terminated instead)
4. Split generation ≤ 5

### Dispersion Vector

The dispersion vector encodes chromatic spread orthogonal to the propagation direction, computed from the difference between violet-edge and center-wavelength refraction directions. On reflection, `last_split` is mirrored across the surface normal. On refraction, it's rotated to the new tangent plane.

### Optimization Strategies (from design doc)

- **Ray reordering** by action type to reduce warp divergence
- **Precomputed IOR LUT** (wavelength → n) as a texture
- **Adaptive split threshold** scaled by screen-space pixel size at hit distance
- **Energy culling** — rays with `length(energy) < 0.01` are terminated
- **Interval merging** — adjacent wavelength ranges with similar directions can be coalesced
- **Double-buffered ray queues** (ping-pong) to avoid per-frame reallocation
- **Ray query object reuse** — single query object per thread

### Scene Configuration XML

Scene files in `assets/` define camera resolution, max bounce depth, obj model placement (scale/rotation/translation), per-object materials (IOR, albedo, diffuse color, shininess), and lights (point, spot, directional, ambient). Parsed via tinyxml2.

## Current Project State

**Phase 1 (Basic Framework)** — not yet started. No application source code exists. The repo currently contains only:
- Design documentation (`docs/`)
- Third-party libraries (`third-party/`)
- Mesh/scene assets (`assets/`)

The first implementation steps are:
1. Create root `CMakeLists.txt` with Vulkan 1.4 + dependencies
2. Set up BLAS/TLAS construction for a simple test scene
3. Implement the compute shader + ray query pipeline
4. Verify monochromatic (non-dispersive) tracing before adding spectral features

## Reference Materials

- `docs/项目设计文档：自适应光谱追踪渲染器.md` — full technical design (data structures, algorithms, optimization strategies, roadmap)
- `docs/项目开发记录文档.md` — development log
- `docs/lost_and_found.md` — SDK/library paths and dependency notes
- Khronos `VK_KHR_ray_query` specification
- NVIDIA Vulkan Ray Tracing Tutorial
- "Ray Tracing Gems II" Chapter 22: Spectral Rendering
