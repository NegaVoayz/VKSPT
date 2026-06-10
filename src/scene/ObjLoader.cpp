#include "scene/ObjLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <filesystem>
#include <stdexcept>

AccelerationStructure::MeshData loadObjMesh(const std::string& objPath) {
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      err;

    // Resolve path relative to assets/
    std::string fullPath = objPath;
    if (!std::filesystem::exists(fullPath)) {
        fullPath = "../../assets/" + objPath;
    }

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                               fullPath.c_str(), "assets/", true);
    if (!ok) {
        throw std::runtime_error(
            "Failed to load OBJ: " + objPath + " — " + err);
    }

    AccelerationStructure::MeshData mesh;
    mesh.vertices = attrib.vertices;
    mesh.normals  = attrib.normals;

    for (const auto& shape : shapes) {
        const auto& m = shape.mesh;
        for (size_t f = 0; f < m.num_face_vertices.size(); ++f) {
            int nv = m.num_face_vertices[f];
            if (nv != 3) {
                throw std::runtime_error(
                    "Non-triangle face in OBJ after triangulation: "
                    + objPath);
            }
            for (int v = 0; v < nv; ++v) {
                const auto& idx =
                    m.indices[f * static_cast<size_t>(nv) + v];
                mesh.indices.push_back(
                    static_cast<uint32_t>(idx.vertex_index));
            }
        }
    }

    // Generate smooth normals if OBJ doesn't provide them
    if (mesh.normals.empty() && !mesh.vertices.empty()
        && !mesh.indices.empty()) {
        size_t numVerts = mesh.vertices.size() / 3;
        mesh.normals.assign(numVerts * 3, 0.0f);

        for (size_t f = 0; f < mesh.indices.size(); f += 3) {
            uint32_t i0 = mesh.indices[f + 0];
            uint32_t i1 = mesh.indices[f + 1];
            uint32_t i2 = mesh.indices[f + 2];
            float* v0 = &mesh.vertices[i0 * 3];
            float* v1 = &mesh.vertices[i1 * 3];
            float* v2 = &mesh.vertices[i2 * 3];
            float e1x = v1[0] - v0[0], e1y = v1[1] - v0[1],
                  e1z = v1[2] - v0[2];
            float e2x = v2[0] - v0[0], e2y = v2[1] - v0[1],
                  e2z = v2[2] - v0[2];
            float nx = e1y * e2z - e1z * e2y;
            float ny = e1z * e2x - e1x * e2z;
            float nz = e1x * e2y - e1y * e2x;
            for (int k = 0; k < 3; ++k) {
                uint32_t idx = mesh.indices[f + k];
                mesh.normals[idx * 3 + 0] += nx;
                mesh.normals[idx * 3 + 1] += ny;
                mesh.normals[idx * 3 + 2] += nz;
            }
        }

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
