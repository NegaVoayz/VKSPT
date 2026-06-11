#pragma once

#include <glm/glm.hpp>

#include <string>
#include <vector>

/// Parsed scene description from XML config + OBJ loading.
/// Owns no GPU resources — just CPU-side data used to build the scene.
struct SceneDescription {
    // ---- Camera ----
    uint32_t    cameraWidth  = 256;
    uint32_t    cameraHeight = 192;
    std::string outputName   = "output.png";
    int         maxDepth     = 4;

    // ---- Object entries (one per XML object element) ----
    enum class MaterialType { Dielectric, Metal, Lambertian, Checkerboard };
    struct MaterialData {
        MaterialType type   = MaterialType::Lambertian;
        float        ior         = 1.0f;              // refractive index (dielectric)
        float        dispersionB = 0.004f;             // Cauchy B for IOR dispersion
        float        roughness   = 1.0f;              // shininess exponent
        glm::vec3    albedo{ 1.0f, 1.0f, 1.0f };     // base color (diffuse/metal)
        glm::vec3    absorbA{ 0.0f, 0.0f, 0.0f };    // Cauchy absorption α(λ)=A+B/λ²
        glm::vec3    absorbB{ 0.0f, 0.0f, 0.0f };
    };
    struct ObjectEntry {
        std::string objFilename;                     // e.g. "duck.obj"
        bool        display           = true;
        bool        normalInterpolation = false;
        glm::vec3   scale{ 1.0f, 1.0f, 1.0f };
        glm::vec3   rotation{ 0.0f, 0.0f, 0.0f };   // Euler angles in degrees
        glm::vec3   translation{ 0.0f, 0.0f, 0.0f };
        MaterialData material;
    };
    std::vector<ObjectEntry> objects;

    // ---- Lights (parsed; per-light shading deferred to future phase) ----
    struct PointLight { glm::vec3 pos, color; float intensity, maxDist; };
    struct SpotLight  { glm::vec3 pos, dir, color; float inner, outer, intensity, maxDist; };
    struct DirLight   { glm::vec3 dir, color; float intensity; };
    struct Ambient    { float strength; glm::vec3 color; };

    std::vector<PointLight> pointLights;
    std::vector<SpotLight>  spotLights;
    std::vector<DirLight>   dirLights;
    Ambient                 ambient{ 0.1f, {0.1f, 0.1f, 0.1f} };
    bool                    envMapDisplay = false;

    // ---- Spheres (legacy: not yet implemented) ----
    bool sphereDisplay[4] = { false };

    // ---- Strength multipliers ----
    float diffuseStrength  = 0.5f;
    float specularStrength = 1.0f;

    // ---- Resource limits (from XML, configurable per-scene) ----
    uint32_t maxInstances = 16;
    uint32_t maxMaterials = 16;
    uint32_t maxLights    = 8;

    // ---- Misc (informational) ----
    int refractionMethod = 1;
    int rayOffsetMethod  = 1;
};

/// Build a 3×4 affine row-major transform matrix from scale, Euler rotation (degrees),
/// and translation.  Result is 3 rows of 4 floats suitable for
/// VkAccelerationStructureInstanceKHR::transform.
void buildTransformMatrix(const glm::vec3& scale,
                          const glm::vec3& rotation,
                          const glm::vec3& translation,
                          float            out[3][4]);
