#pragma once

#include "scene/SceneConfig.h"
#include <string>

/// Parse a SceneInfo XML file into a SceneDescription.
SceneDescription ParseSceneXML(const std::string& xmlPath);
