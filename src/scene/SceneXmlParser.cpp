#include "scene/SceneXmlParser.h"

#include <tinyxml2.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <unordered_set>

// XML element tags that are NOT objects (parsed separately elsewhere)
static const std::unordered_set<std::string> kNonObjectTags = {
    "RefractionMethod", "EnvironmentMap", "DepthMax", "Camera",
    "SpheresDisplay", "PointLight", "PointLight2", "SpotLight",
    "DirectionalLight", "AmbientAgrs", "Strength", "ResourceLimits",
    "RayOffsetMethod"
};

static float xmlFloat(tinyxml2::XMLElement* el, const char* attr, float fallback = 0.0f)
{ return el ? el->FloatAttribute(attr, fallback) : fallback; }

static int xmlInt(tinyxml2::XMLElement* el, const char* attr, int fallback = 0)
{ return el ? el->IntAttribute(attr, fallback) : fallback; }

static const char* xmlStr(tinyxml2::XMLElement* el, const char* attr, const char* fallback = "")
{
    if (!el) return fallback;
    const char* v = el->Attribute(attr);
    return v ? v : fallback;
}

static SceneDescription::ObjectEntry parseObject(tinyxml2::XMLElement* objEl)
{
    SceneDescription::ObjectEntry entry;
    if (auto* a = objEl->FirstChildElement("Args")) {
        entry.objFilename = xmlStr(a, "filename");
        entry.display = (xmlInt(a, "display", 1) != 0);
        entry.normalInterpolation = (xmlInt(a, "normalinterpolation", 0) != 0);
    }
    if (auto* s = objEl->FirstChildElement("Scale"))
        entry.scale = {xmlFloat(s,"x",1), xmlFloat(s,"y",1), xmlFloat(s,"z",1)};
    if (auto* r = objEl->FirstChildElement("Rotation"))
        entry.rotation = {xmlFloat(r,"x",0), xmlFloat(r,"y",0), xmlFloat(r,"z",0)};
    if (auto* t = objEl->FirstChildElement("Translation"))
        entry.translation = {xmlFloat(t,"x",0), xmlFloat(t,"y",0), xmlFloat(t,"z",0)};

    if (auto* matEl = objEl->FirstChildElement("Material")) {
        auto& mat = entry.material;
        const char* tstr = xmlStr(matEl, "type", "lambertian");
        if (std::strcmp(tstr, "dielectric") == 0)
            mat.type = SceneDescription::MaterialType::Dielectric;
        else if (std::strcmp(tstr, "metal") == 0)
            mat.type = SceneDescription::MaterialType::Metal;
        else if (std::strcmp(tstr, "checkerboard") == 0)
            mat.type = SceneDescription::MaterialType::Checkerboard;
        else
            mat.type = SceneDescription::MaterialType::Lambertian;

        if (auto* da = matEl->FirstChildElement("DispersionA"))
            mat.ior = xmlFloat(da, "value", 1.0f);
        if (auto* db = matEl->FirstChildElement("DispersionB"))
            mat.dispersionB = xmlFloat(db, "value", 0.004f);
        if (auto* ro = matEl->FirstChildElement("Roughness"))
            mat.roughness = xmlFloat(ro, "value", 1.0f);
        if (auto* al = matEl->FirstChildElement("Albedo"))
            mat.albedo = {xmlFloat(al,"r",1), xmlFloat(al,"g",1), xmlFloat(al,"b",1)};
        if (auto* aa = matEl->FirstChildElement("AbsorptionA"))
            mat.absorbA = xmlFloat(aa, "value", 0.0f);
        if (auto* ab = matEl->FirstChildElement("AbsorptionB"))
            mat.absorbB = xmlFloat(ab, "value", 0.0f);
        if (auto* rf = matEl->FirstChildElement("Reflectivity"))
            mat.reflectivity = xmlFloat(rf, "value", 1.0f);
    }
    return entry;
}

static void parseLights(tinyxml2::XMLElement* root, SceneDescription& desc)
{
    for (const char* tag : {"PointLight", "PointLight2"}) {
        for (auto* el = root->FirstChildElement(tag); el;
             el = el->NextSiblingElement(tag)) {
            SceneDescription::PointLight pl;
            if (auto* p = el->FirstChildElement("Position"))
                pl.pos = {xmlFloat(p,"x"), xmlFloat(p,"y"), xmlFloat(p,"z")};
            if (auto* c = el->FirstChildElement("Color"))
                pl.color = {xmlFloat(c,"r"), xmlFloat(c,"g"), xmlFloat(c,"b")};
            if (auto* a = el->FirstChildElement("Args")) {
                pl.intensity = xmlFloat(a, "intensity", 1.0f);
                pl.maxDist   = xmlFloat(a, "maxDistance", 100.0f);
            }
            desc.pointLights.push_back(pl);
        }
    }
    for (auto* el = root->FirstChildElement("SpotLight"); el;
         el = el->NextSiblingElement("SpotLight")) {
        SceneDescription::SpotLight sl;
        if (auto* p = el->FirstChildElement("Position"))
            sl.pos = {xmlFloat(p,"x"), xmlFloat(p,"y"), xmlFloat(p,"z")};
        if (auto* d = el->FirstChildElement("Direction"))
            sl.dir = {xmlFloat(d,"x"), xmlFloat(d,"y"), xmlFloat(d,"z")};
        if (auto* c = el->FirstChildElement("Color"))
            sl.color = {xmlFloat(c,"r"), xmlFloat(c,"g"), xmlFloat(c,"b")};
        if (auto* a = el->FirstChildElement("Args")) {
            sl.inner = xmlFloat(a, "innerAngle", 15.0f);
            sl.outer = xmlFloat(a, "outerAngle", 30.0f);
            sl.intensity = xmlFloat(a, "intensity", 1.0f);
            sl.maxDist   = xmlFloat(a, "maxDistance", 100.0f);
        }
        desc.spotLights.push_back(sl);
    }
    for (auto* el = root->FirstChildElement("DirectionalLight"); el;
         el = el->NextSiblingElement("DirectionalLight")) {
        SceneDescription::DirLight dl;
        if (auto* d = el->FirstChildElement("Direction"))
            dl.dir = {xmlFloat(d,"x"), xmlFloat(d,"y"), xmlFloat(d,"z")};
        if (auto* c = el->FirstChildElement("Color"))
            dl.color = {xmlFloat(c,"r"), xmlFloat(c,"g"), xmlFloat(c,"b")};
        if (auto* i = el->FirstChildElement("Intensity"))
            dl.intensity = xmlFloat(i, "value", 1.0f);
        desc.dirLights.push_back(dl);
    }
    if (auto* amb = root->FirstChildElement("AmbientAgrs")) {
        if (auto* s = amb->FirstChildElement("Strength"))
            desc.ambient.strength = xmlFloat(s, "value", 0.1f);
        if (auto* c = amb->FirstChildElement("Color"))
            desc.ambient.color = {
                xmlFloat(c,"r",0.1f), xmlFloat(c,"g",0.1f), xmlFloat(c,"b",0.1f)};
    }
}

static void parseGlobalSettings(tinyxml2::XMLElement* root, SceneDescription& desc)
{
    if (auto* cam = root->FirstChildElement("Camera"))
        if (auto* a = cam->FirstChildElement("Args")) {
            desc.cameraWidth  = xmlInt(a, "width", 256);
            desc.cameraHeight = xmlInt(a, "height", 192);
            desc.outputName   = xmlStr(a, "outputname", "output.png");
        }
    if (auto* dm = root->FirstChildElement("DepthMax"))
        if (auto* a = dm->FirstChildElement("Args"))
            desc.maxDepth = xmlInt(a, "DepthMax", 4);
    if (auto* rm = root->FirstChildElement("RefractionMethod"))
        if (auto* a = rm->FirstChildElement("Args"))
            desc.refractionMethod = xmlInt(a, "version", 1);
    if (auto* rom = root->FirstChildElement("RayOffsetMethod"))
        if (auto* a = rom->FirstChildElement("Args"))
            desc.rayOffsetMethod = xmlInt(a, "version", 1);
    if (auto* em = root->FirstChildElement("EnvironmentMap"))
        if (auto* a = em->FirstChildElement("Args"))
            desc.envMapDisplay = (xmlInt(a, "display", 0) != 0);
    if (auto* rl = root->FirstChildElement("ResourceLimits"))
        if (auto* a = rl->FirstChildElement("Args")) {
            desc.maxInstances = xmlInt(a, "maxInstances", 16);
            desc.maxMaterials = xmlInt(a, "maxMaterials", 16);
            desc.maxLights    = xmlInt(a, "maxLights", 8);
        }
    if (auto* str = root->FirstChildElement("Strength")) {
        if (auto* d = str->FirstChildElement("diffuseStrength"))
            desc.diffuseStrength = xmlFloat(d, "value", 0.5f);
        if (auto* s = str->FirstChildElement("specularStrength"))
            desc.specularStrength = xmlFloat(s, "value", 1.0f);
    }
    if (auto* sd = root->FirstChildElement("SpheresDisplay"))
        if (auto* a = sd->FirstChildElement("Args")) {
            desc.sphereDisplay[0] = (xmlInt(a,"sphere1",0) != 0);
            desc.sphereDisplay[1] = (xmlInt(a,"sphere2",0) != 0);
            desc.sphereDisplay[2] = (xmlInt(a,"sphere3",0) != 0);
            desc.sphereDisplay[3] = (xmlInt(a,"sphere4",0) != 0);
        }
}

SceneDescription ParseSceneXML(const std::string& xmlPath) {
    SceneDescription desc;
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(xmlPath.c_str()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("Failed to load XML: " + xmlPath);
    auto* root = doc.FirstChildElement("SceneInfo");
    if (!root)
        throw std::runtime_error("Missing <SceneInfo> in: " + xmlPath);

    parseGlobalSettings(root, desc);

    for (auto* obj = root->FirstChildElement(); obj;
         obj = obj->NextSiblingElement()) {
        const char* tag = obj->Name();
        if (!tag || kNonObjectTags.count(tag)) continue;
        auto e = parseObject(obj);
        if (e.display && !e.objFilename.empty())
            desc.objects.push_back(std::move(e));
    }

    parseLights(root, desc);
    return desc;
}
