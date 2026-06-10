#pragma once

#include "ray/AccelerationStructure.h"
#include "scene/SceneConfig.h"

/// Builds GPU scene data from a parsed SceneDescription.
class SceneBuilder {
public:
    void build(const SceneDescription& desc,
               AccelerationStructure& as);
};
