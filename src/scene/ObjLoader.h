#pragma once

#include "ray/AccelerationStructure.h"
#include <cstdint>
#include <string>
#include <vector>

/// Load an OBJ mesh from disk. Returns MeshData with vertices, indices,
/// and per-vertex normals (generated if the OBJ doesn't provide them).
AccelerationStructure::MeshData loadObjMesh(const std::string& objPath);
