#include "SceneConfig.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <tinyxml2.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>

// =============================================================================
// OBJ Loading
// =============================================================================

AccelerationStructure::MeshData loadObjMesh(const std::string& objPath) {
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      err;

    // Resolve path relative to assets/ (running from build/Debug, so use ../../assets/)
    std::string fullPath = objPath;
    if (!std::filesystem::exists(fullPath)) {
        fullPath = "../../assets/" + objPath;
    }

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                               fullPath.c_str(), "assets/", true);
    if (!ok) {
        throw std::runtime_error("Failed to load OBJ: " + objPath + " — " + err);
    }

    AccelerationStructure::MeshData mesh;

    // Keep the indexed representation: vertices as-is (xyz interleaved), indices flattened.
    mesh.vertices = attrib.vertices;   // already interleaved xyz floats
    mesh.normals  = attrib.normals;    // interleaved xyz normals (same indexing; empty if no vn lines)

    for (const auto& shape : shapes) {
        const auto& m = shape.mesh;
        for (size_t f = 0; f < m.num_face_vertices.size(); ++f) {
            // After triangulation, all faces have 3 vertices
            int nv = m.num_face_vertices[f];
            if (nv != 3) {
                throw std::runtime_error(
                    "Non-triangle face in OBJ after triangulation: " + objPath);
            }
            for (int v = 0; v < nv; ++v) {
                const auto& idx = m.indices[f * size_t(nv) + v];
                mesh.indices.push_back(static_cast<uint32_t>(idx.vertex_index));
            }
        }
    }

    // If the OBJ doesn't have vertex normals, generate smooth normals from geometry
    if (mesh.normals.empty() && !mesh.vertices.empty() && !mesh.indices.empty()) {
        size_t numVerts = mesh.vertices.size() / 3;
        mesh.normals.assign(numVerts * 3, 0.0f);

        for (size_t f = 0; f < mesh.indices.size(); f += 3) {
            uint32_t i0 = mesh.indices[f + 0];
            uint32_t i1 = mesh.indices[f + 1];
            uint32_t i2 = mesh.indices[f + 2];

            // Read vertex positions
            float* v0 = &mesh.vertices[i0 * 3];
            float* v1 = &mesh.vertices[i1 * 3];
            float* v2 = &mesh.vertices[i2 * 3];

            // Compute face normal (cross product of edges, area-weighted)
            float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
            float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];
            float nx = e1y * e2z - e1z * e2y;
            float ny = e1z * e2x - e1x * e2z;
            float nz = e1x * e2y - e1y * e2x;

            // Accumulate to each vertex of the face
            for (int k = 0; k < 3; ++k) {
                uint32_t idx = mesh.indices[f + k];
                mesh.normals[idx * 3 + 0] += nx;
                mesh.normals[idx * 3 + 1] += ny;
                mesh.normals[idx * 3 + 2] += nz;
            }
        }

        // Normalize all vertex normals
        for (size_t i = 0; i < numVerts; ++i) {
            float x = mesh.normals[i * 3 + 0];
            float y = mesh.normals[i * 3 + 1];
            float z = mesh.normals[i * 3 + 2];
            float len = std::sqrt(x * x + y * y + z * z);
            if (len > 1e-8f) {
                mesh.normals[i * 3 + 0] /= len;
                mesh.normals[i * 3 + 1] /= len;
                mesh.normals[i * 3 + 2] /= len;
            }
        }
    }

    return mesh;
}

// =============================================================================
// Transform matrix
// =============================================================================

void buildTransformMatrix(const glm::vec3& scale,
                          const glm::vec3& rotation,
                          const glm::vec3& translation,
                          float            out[3][4])
{
    glm::mat4 m(1.0f);
    m = glm::translate(m, translation);
    m = glm::rotate(m, glm::radians(rotation.z), glm::vec3(0, 0, 1));
    m = glm::rotate(m, glm::radians(rotation.y), glm::vec3(0, 1, 0));
    m = glm::rotate(m, glm::radians(rotation.x), glm::vec3(1, 0, 0));
    m = glm::scale(m, scale);

    // Vulkan VkTransformMatrixKHR is row-major 3×4 (the last row is implicitly (0,0,0,1))
    // glm uses column-major storage, so we transpose when extracting rows.
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[r][c] = m[c][r];   // column-major → row-major
        }
    }
}

// =============================================================================
// XML Parsing
// =============================================================================

// Known object element names in the XML schema
static const char* kObjectNames[] = {
    "Duck", "Bunny", "Dragon", "Venus", "Asschercut", "FudanLogo", "Prism", "Checkerboard"
};
static constexpr int kNumObjectNames = 8;

/// Read a float attribute from an XMLElement, with fallback value.
static float xmlFloat(tinyxml2::XMLElement* el, const char* attr, float fallback = 0.0f) {
    if (!el) return fallback;
    return el->FloatAttribute(attr, fallback);
}

/// Read an int attribute from an XMLElement, with fallback value.
static int xmlInt(tinyxml2::XMLElement* el, const char* attr, int fallback = 0) {
    if (!el) return fallback;
    return el->IntAttribute(attr, fallback);
}

/// Read a string attribute, with fallback.
static const char* xmlStr(tinyxml2::XMLElement* el, const char* attr,
                          const char* fallback = "") {
    if (!el) return fallback;
    const char* v = el->Attribute(attr);
    return v ? v : fallback;
}

/// Parse one object element (Duck, Bunny, Dragon, etc.) into ObjectEntry.
static SceneDescription::ObjectEntry parseObject(tinyxml2::XMLElement* objEl) {
    SceneDescription::ObjectEntry entry;

    // <Args> — filename, display, normalinterpolation, rendermethod
    if (auto* args = objEl->FirstChildElement("Args")) {
        entry.objFilename         = xmlStr(args, "filename");
        entry.display             = (xmlInt(args, "display", 1) != 0);
        entry.normalInterpolation = (xmlInt(args, "normalinterpolation", 0) != 0);
    }

    // <Scale>
    if (auto* s = objEl->FirstChildElement("Scale")) {
        entry.scale.x = xmlFloat(s, "x", 1.0f);
        entry.scale.y = xmlFloat(s, "y", 1.0f);
        entry.scale.z = xmlFloat(s, "z", 1.0f);
    }

    // <Rotation>
    if (auto* r = objEl->FirstChildElement("Rotation")) {
        entry.rotation.x = xmlFloat(r, "x", 0.0f);
        entry.rotation.y = xmlFloat(r, "y", 0.0f);
        entry.rotation.z = xmlFloat(r, "z", 0.0f);
    }

    // <Translation>
    if (auto* t = objEl->FirstChildElement("Translation")) {
        entry.translation.x = xmlFloat(t, "x", 0.0f);
        entry.translation.y = xmlFloat(t, "y", 0.0f);
        entry.translation.z = xmlFloat(t, "z", 0.0f);
    }

    // <RefractiveIndex>
    if (auto* ri = objEl->FirstChildElement("RefractiveIndex")) {
        entry.ior = xmlFloat(ri, "x", 1.0f);
    }

    // <Albedo>
    if (auto* al = objEl->FirstChildElement("Albedo")) {
        entry.albedoWt.x = xmlFloat(al, "x", 0.0f);
        entry.albedoWt.y = xmlFloat(al, "y", 0.0f);
        entry.albedoWt.z = xmlFloat(al, "z", 0.0f);
        entry.albedoWt.w = xmlFloat(al, "w", 0.0f);
    }

    // <Diffuse>
    if (auto* d = objEl->FirstChildElement("Diffuse")) {
        entry.diffuse.r = xmlFloat(d, "r", 1.0f);
        entry.diffuse.g = xmlFloat(d, "g", 1.0f);
        entry.diffuse.b = xmlFloat(d, "b", 1.0f);
    }

    // <Shiness>  (note: XML uses "Shiness" not "Shininess")
    if (auto* sh = objEl->FirstChildElement("Shiness")) {
        entry.shininess = xmlFloat(sh, "p", 1.0f);
    }

    // <AbsorptionA> — Cauchy A coefficient (α = A + B/λ²), RGB
    if (auto* aa = objEl->FirstChildElement("AbsorptionA")) {
        entry.absorbA.x = xmlFloat(aa, "x", 0.0f);
        entry.absorbA.y = xmlFloat(aa, "y", 0.0f);
        entry.absorbA.z = xmlFloat(aa, "z", 0.0f);
    }

    // <AbsorptionB> — Cauchy B coefficient (α = A + B/λ²), RGB
    if (auto* ab = objEl->FirstChildElement("AbsorptionB")) {
        entry.absorbB.x = xmlFloat(ab, "x", 0.0f);
        entry.absorbB.y = xmlFloat(ab, "y", 0.0f);
        entry.absorbB.z = xmlFloat(ab, "z", 0.0f);
    }

    // <DispersionB> — Cauchy B for IOR dispersion n(λ)=A+B/λ²
    if (auto* db = objEl->FirstChildElement("DispersionB")) {
        entry.dispersionB = xmlFloat(db, "value", 0.004f);
    }

    return entry;
}

/// Parse a light element (PointLight, PointLight2, SpotLight, DirectionalLight)
/// and update the scene description.
static void parseLights(tinyxml2::XMLElement* sceneRoot,
                        SceneDescription& desc)
{
    // PointLight or PointLight2
    for (const char* tag : {"PointLight", "PointLight2"}) {
        for (auto* el = sceneRoot->FirstChildElement(tag); el;
             el = el->NextSiblingElement(tag))
        {
            SceneDescription::PointLight pl;
            if (auto* p = el->FirstChildElement("Position")) {
                pl.pos.x = xmlFloat(p, "x"); pl.pos.y = xmlFloat(p, "y"); pl.pos.z = xmlFloat(p, "z");
            }
            if (auto* c = el->FirstChildElement("Color")) {
                pl.color.r = xmlFloat(c, "r"); pl.color.g = xmlFloat(c, "g"); pl.color.b = xmlFloat(c, "b");
            }
            if (auto* a = el->FirstChildElement("Args")) {
                pl.intensity = xmlFloat(a, "intensity", 1.0f);
                pl.maxDist   = xmlFloat(a, "maxDistance", 100.0f);
            }
            desc.pointLights.push_back(pl);
        }
    }

    // SpotLight
    for (auto* el = sceneRoot->FirstChildElement("SpotLight"); el;
         el = el->NextSiblingElement("SpotLight"))
    {
        SceneDescription::SpotLight sl;
        if (auto* p = el->FirstChildElement("Position")) {
            sl.pos.x = xmlFloat(p, "x"); sl.pos.y = xmlFloat(p, "y"); sl.pos.z = xmlFloat(p, "z");
        }
        if (auto* d = el->FirstChildElement("Direction")) {
            sl.dir.x = xmlFloat(d, "x"); sl.dir.y = xmlFloat(d, "y"); sl.dir.z = xmlFloat(d, "z");
        }
        if (auto* c = el->FirstChildElement("Color")) {
            sl.color.r = xmlFloat(c, "r"); sl.color.g = xmlFloat(c, "g"); sl.color.b = xmlFloat(c, "b");
        }
        if (auto* a = el->FirstChildElement("Args")) {
            sl.inner     = xmlFloat(a, "innerAngle", 15.0f);
            sl.outer     = xmlFloat(a, "outerAngle", 30.0f);
            sl.intensity = xmlFloat(a, "intensity", 1.0f);
            sl.maxDist   = xmlFloat(a, "maxDistance", 100.0f);
        }
        desc.spotLights.push_back(sl);
    }

    // DirectionalLight
    for (auto* el = sceneRoot->FirstChildElement("DirectionalLight"); el;
         el = el->NextSiblingElement("DirectionalLight"))
    {
        SceneDescription::DirLight dl;
        if (auto* d = el->FirstChildElement("Direction")) {
            dl.dir.x = xmlFloat(d, "x"); dl.dir.y = xmlFloat(d, "y"); dl.dir.z = xmlFloat(d, "z");
        }
        if (auto* c = el->FirstChildElement("Color")) {
            dl.color.r = xmlFloat(c, "r"); dl.color.g = xmlFloat(c, "g"); dl.color.b = xmlFloat(c, "b");
        }
        if (auto* i = el->FirstChildElement("Intensity")) {
            dl.intensity = xmlFloat(i, "value", 1.0f);
        }
        desc.dirLights.push_back(dl);
    }

    // Ambient
    if (auto* amb = sceneRoot->FirstChildElement("AmbientAgrs")) {
        if (auto* s = amb->FirstChildElement("Strength")) {
            desc.ambient.strength = xmlFloat(s, "value", 0.1f);
        }
        if (auto* c = amb->FirstChildElement("Color")) {
            desc.ambient.color.r = xmlFloat(c, "r", 0.1f);
            desc.ambient.color.g = xmlFloat(c, "g", 0.1f);
            desc.ambient.color.b = xmlFloat(c, "b", 0.1f);
        }
    }
}

SceneDescription parseSceneXML(const std::string& xmlPath) {
    SceneDescription desc;

    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError err = doc.LoadFile(xmlPath.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("Failed to load XML: " + xmlPath);
    }

    auto* root = doc.FirstChildElement("SceneInfo");
    if (!root) {
        throw std::runtime_error("Missing <SceneInfo> root element in: " + xmlPath);
    }

    // ---- Camera ----
    if (auto* cam = root->FirstChildElement("Camera")) {
        if (auto* args = cam->FirstChildElement("Args")) {
            desc.cameraWidth  = xmlInt(args, "width", 256);
            desc.cameraHeight = xmlInt(args, "height", 192);
            desc.outputName   = xmlStr(args, "outputname", "output.png");
        }
    }

    // ---- DepthMax ----
    if (auto* dm = root->FirstChildElement("DepthMax")) {
        if (auto* args = dm->FirstChildElement("Args")) {
            desc.maxDepth = xmlInt(args, "DepthMax", 4);
        }
    }

    // ---- RefractionMethod ----
    if (auto* rm = root->FirstChildElement("RefractionMethod")) {
        if (auto* args = rm->FirstChildElement("Args")) {
            desc.refractionMethod = xmlInt(args, "version", 1);
        }
    }

    // ---- RayOffsetMethod ----
    if (auto* rom = root->FirstChildElement("RayOffsetMethod")) {
        if (auto* args = rom->FirstChildElement("Args")) {
            desc.rayOffsetMethod = xmlInt(args, "version", 1);
        }
    }

    // ---- EnvironmentMap ----
    if (auto* em = root->FirstChildElement("EnvironmentMap")) {
        if (auto* args = em->FirstChildElement("Args")) {
            desc.envMapDisplay = (xmlInt(args, "display", 0) != 0);
        }
    }

    // ---- SpheresDisplay ----
    if (auto* sd = root->FirstChildElement("SpheresDisplay")) {
        if (auto* args = sd->FirstChildElement("Args")) {
            desc.sphereDisplay[0] = (xmlInt(args, "sphere1", 0) != 0);
            desc.sphereDisplay[1] = (xmlInt(args, "sphere2", 0) != 0);
            desc.sphereDisplay[2] = (xmlInt(args, "sphere3", 0) != 0);
            desc.sphereDisplay[3] = (xmlInt(args, "sphere4", 0) != 0);
        }
    }

    // ---- Object elements ----
    for (int i = 0; i < kNumObjectNames; ++i) {
        for (auto* objEl = root->FirstChildElement(kObjectNames[i]); objEl;
             objEl = objEl->NextSiblingElement(kObjectNames[i]))
        {
            auto entry = parseObject(objEl);
            if (entry.display && !entry.objFilename.empty()) {
                desc.objects.push_back(std::move(entry));
            }
        }
    }

    // ---- Lights ----
    parseLights(root, desc);

    return desc;
}
