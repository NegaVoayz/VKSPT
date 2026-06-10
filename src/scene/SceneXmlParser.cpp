#include "scene/SceneXmlParser.h"

#include <tinyxml2.h>
#include <stdexcept>
#include <cstring>

// Known object element names in the XML schema
static const char* kObjectNames[] = {
    "Duck","Bunny","Dragon","Venus","Asschercut","FudanLogo",
    "Prism","Checkerboard","Box"
};
static constexpr int kNumObjectNames = 9;

static float xmlFloat(tinyxml2::XMLElement* el,
    const char* attr, float fallback = 0.0f)
{
    if (!el) return fallback;
    return el->FloatAttribute(attr, fallback);
}

static int xmlInt(tinyxml2::XMLElement* el,
    const char* attr, int fallback = 0)
{
    if (!el) return fallback;
    return el->IntAttribute(attr, fallback);
}

static const char* xmlStr(tinyxml2::XMLElement* el,
    const char* attr, const char* fallback = "")
{
    if (!el) return fallback;
    const char* v = el->Attribute(attr);
    return v ? v : fallback;
}

static SceneDescription::ObjectEntry parseObject(
    tinyxml2::XMLElement* objEl)
{
    SceneDescription::ObjectEntry entry;
    if (auto* a = objEl->FirstChildElement("Args")) {
        entry.objFilename = xmlStr(a, "filename");
        entry.display = (xmlInt(a, "display", 1) != 0);
        entry.normalInterpolation =
            (xmlInt(a, "normalinterpolation", 0) != 0);
    }
    if (auto* s = objEl->FirstChildElement("Scale")) {
        entry.scale = {xmlFloat(s,"x",1), xmlFloat(s,"y",1), xmlFloat(s,"z",1)};
    }
    if (auto* r = objEl->FirstChildElement("Rotation")) {
        entry.rotation = {xmlFloat(r,"x",0), xmlFloat(r,"y",0), xmlFloat(r,"z",0)};
    }
    if (auto* t = objEl->FirstChildElement("Translation")) {
        entry.translation = {xmlFloat(t,"x",0), xmlFloat(t,"y",0), xmlFloat(t,"z",0)};
    }
    if (auto* ri = objEl->FirstChildElement("RefractiveIndex"))
        entry.ior = xmlFloat(ri, "x", 1.0f);
    if (auto* al = objEl->FirstChildElement("Albedo")) {
        entry.albedoWt = {xmlFloat(al,"x",0), xmlFloat(al,"y",0),
                          xmlFloat(al,"z",0), xmlFloat(al,"w",0)};
    }
    if (auto* d = objEl->FirstChildElement("Diffuse")) {
        entry.diffuse = {xmlFloat(d,"r",1), xmlFloat(d,"g",1), xmlFloat(d,"b",1)};
    }
    if (auto* sh = objEl->FirstChildElement("Shiness"))
        entry.shininess = xmlFloat(sh, "p", 1.0f);
    if (auto* aa = objEl->FirstChildElement("AbsorptionA")) {
        entry.absorbA = {xmlFloat(aa,"x",0), xmlFloat(aa,"y",0), xmlFloat(aa,"z",0)};
    }
    if (auto* ab = objEl->FirstChildElement("AbsorptionB")) {
        entry.absorbB = {xmlFloat(ab,"x",0), xmlFloat(ab,"y",0), xmlFloat(ab,"z",0)};
    }
    if (auto* db = objEl->FirstChildElement("DispersionB"))
        entry.dispersionB = xmlFloat(db, "value", 0.004f);
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

SceneDescription parseSceneXML(const std::string& xmlPath) {
    SceneDescription desc;
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(xmlPath.c_str()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("Failed to load XML: " + xmlPath);
    auto* root = doc.FirstChildElement("SceneInfo");
    if (!root)
        throw std::runtime_error("Missing <SceneInfo> in: " + xmlPath);

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
    if (auto* sd = root->FirstChildElement("SpheresDisplay"))
        if (auto* a = sd->FirstChildElement("Args")) {
            desc.sphereDisplay[0] = (xmlInt(a,"sphere1",0) != 0);
            desc.sphereDisplay[1] = (xmlInt(a,"sphere2",0) != 0);
            desc.sphereDisplay[2] = (xmlInt(a,"sphere3",0) != 0);
            desc.sphereDisplay[3] = (xmlInt(a,"sphere4",0) != 0);
        }

    for (int i = 0; i < kNumObjectNames; ++i)
        for (auto* obj = root->FirstChildElement(kObjectNames[i]); obj;
             obj = obj->NextSiblingElement(kObjectNames[i])) {
            auto e = parseObject(obj);
            if (e.display && !e.objFilename.empty())
                desc.objects.push_back(std::move(e));
        }

    parseLights(root, desc);
    return desc;
}
