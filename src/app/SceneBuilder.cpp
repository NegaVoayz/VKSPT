#include "app/SceneBuilder.h"
#include "scene/ObjLoader.h"
#include <algorithm>
#include <iostream>

void SceneBuilder::build(const SceneDescription& desc,
                          AccelerationStructure& as)
{
    std::vector<AccelerationStructure::InstanceInfo> instances;
    std::vector<AccelerationStructure::MaterialGPU>  materials;

    for (size_t i = 0; i < desc.objects.size(); ++i) {
        const auto& obj = desc.objects[i];
        std::cout << "  Loading: " << obj.objFilename << std::endl;
        auto mesh = loadObjMesh(obj.objFilename);
        std::cout << "    vertices: " << mesh.vertices.size()/3
                  << ", triangles: " << mesh.indices.size()/3 << std::endl;

        float xf[3][4];
        buildTransformMatrix(obj.scale, obj.rotation, obj.translation, xf);

        AccelerationStructure::InstanceInfo inst;
        inst.mesh = std::move(mesh);
        inst.customIndex = static_cast<uint32_t>(i);
        inst.materialID  = static_cast<uint32_t>(i);
        inst.hasNormals  = obj.normalInterpolation
                        && !inst.mesh.normals.empty();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                inst.transform[r][c] = xf[r][c];
        instances.push_back(std::move(inst));

        AccelerationStructure::MaterialGPU mat{};
        if (obj.ior <= 0.0f) {
            mat.albedo[0]=obj.diffuse.r; mat.albedo[1]=obj.diffuse.g;
            mat.albedo[2]=obj.diffuse.b;
            mat.params[0]=1; mat.params[1]=1/std::max(obj.shininess,1.f);
            mat.params[2]=1;
        } else if (obj.ior > 1.01f) {
            for (int c=0;c<3;++c) {
                mat.cauchyA[c]=obj.ior; mat.cauchyB[c]=obj.dispersionB;
                mat.absorpA[c]=obj.absorbA[c]; mat.absorpB[c]=obj.absorbB[c];
            }
            mat.params[0]=obj.ior; mat.params[1]=0; mat.params[2]=0;
        } else if (obj.objFilename.find("checkerboard")!=std::string::npos) {
            mat.albedo[0]=obj.diffuse.r; mat.albedo[1]=obj.diffuse.g;
            mat.albedo[2]=obj.diffuse.b;
            mat.params[0]=1; mat.params[1]=1/std::max(obj.shininess,1.f);
            mat.params[2]=3;
        } else {
            mat.albedo[0]=obj.diffuse.r; mat.albedo[1]=obj.diffuse.g;
            mat.albedo[2]=obj.diffuse.b;
            mat.params[0]=1; mat.params[1]=1/std::max(obj.shininess,1.f);
            mat.params[2]=2;
        }
        materials.push_back(mat);
    }

    constexpr uint32_t MAX_LIGHTS = 4;
    std::vector<AccelerationStructure::GpuLight> gpuLights;
    {
        AccelerationStructure::GpuLight amb;
        float as = desc.ambient.strength;
        amb.pos_type[0]=0; amb.pos_type[1]=0; amb.pos_type[2]=0;
        amb.pos_type[3]=3;
        amb.color_intensity[0]=desc.ambient.color.r;
        amb.color_intensity[1]=desc.ambient.color.g;
        amb.color_intensity[2]=desc.ambient.color.b;
        amb.color_intensity[3]=as;
        gpuLights.push_back(amb);
    }
    for (const auto& pl : desc.pointLights) {
        if (gpuLights.size()>=MAX_LIGHTS) break;
        AccelerationStructure::GpuLight l{};
        l.pos_type[0]=pl.pos.x; l.pos_type[1]=pl.pos.y;
        l.pos_type[2]=pl.pos.z; l.pos_type[3]=0;
        l.color_intensity[0]=pl.color.r;
        l.color_intensity[1]=pl.color.g;
        l.color_intensity[2]=pl.color.b;
        l.color_intensity[3]=pl.intensity;
        l.outer_range[1]=pl.maxDist;
        gpuLights.push_back(l);
    }
    for (const auto& sl : desc.spotLights) {
        if (gpuLights.size()>=MAX_LIGHTS) break;
        AccelerationStructure::GpuLight l{};
        l.pos_type[0]=sl.pos.x; l.pos_type[1]=sl.pos.y;
        l.pos_type[2]=sl.pos.z; l.pos_type[3]=2;
        l.color_intensity[0]=sl.color.r;
        l.color_intensity[1]=sl.color.g;
        l.color_intensity[2]=sl.color.b;
        l.color_intensity[3]=sl.intensity;
        l.dir_inner[0]=sl.dir.x; l.dir_inner[1]=sl.dir.y;
        l.dir_inner[2]=sl.dir.z; l.dir_inner[3]=sl.inner;
        l.outer_range[0]=sl.outer; l.outer_range[1]=sl.maxDist;
        gpuLights.push_back(l);
    }
    for (const auto& dl : desc.dirLights) {
        if (gpuLights.size()>=MAX_LIGHTS) break;
        AccelerationStructure::GpuLight l{};
        l.pos_type[0]=dl.dir.x; l.pos_type[1]=dl.dir.y;
        l.pos_type[2]=dl.dir.z; l.pos_type[3]=1;
        l.color_intensity[0]=dl.color.r;
        l.color_intensity[1]=dl.color.g;
        l.color_intensity[2]=dl.color.b;
        l.color_intensity[3]=dl.intensity;
        gpuLights.push_back(l);
    }
    while (gpuLights.size() < MAX_LIGHTS)
        gpuLights.push_back(AccelerationStructure::GpuLight{});

    as.buildScene(instances, materials, gpuLights);
}
