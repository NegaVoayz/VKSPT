#include "app/SceneBuilder.h"
#include "core/Log.h"
#include "scene/ObjLoader.h"
#include <algorithm>
#include <cmath>

namespace {

void buildInstances(const SceneDescription& desc,
                    AccelerationStructure& as,
                    std::vector<AccelerationStructure::InstanceInfo>& instances,
                    std::vector<AccelerationStructure::MaterialGPU>& materials)
{
    for (size_t i = 0; i < desc.objects.size(); ++i) {
        const auto& obj = desc.objects[i];
        Log::info("  Loading: {}", obj.objFilename);
        auto mesh = loadObjMesh(obj.objFilename);
        Log::info("    vertices: {}, triangles: {}",
                  mesh.vertices.size()/3, mesh.indices.size()/3);

        float xf[3][4];
        BuildTransformMatrix(obj.scale, obj.rotation, obj.translation, xf);

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
        auto& m = obj.material;
        switch (m.type) {
        case SceneDescription::MaterialType::Dielectric:
            for (int c=0;c<3;++c) {
                mat.cauchyA[c]=m.ior; mat.cauchyB[c]=m.dispersionB;
                mat.absorpA[c]=m.absorbA[c]; mat.absorpB[c]=m.absorbB[c];
            }
            mat.params[0]=m.ior; mat.params[1]=0; mat.params[2]=0;
            break;
        case SceneDescription::MaterialType::Metal:
            mat.albedo[0]=m.albedo.r; mat.albedo[1]=m.albedo.g;
            mat.albedo[2]=m.albedo.b;
            mat.params[0]=1; mat.params[1]=1/std::max(m.roughness,1.f);
            mat.params[2]=1;
            break;
        case SceneDescription::MaterialType::Checkerboard:
            mat.albedo[0]=m.albedo.r; mat.albedo[1]=m.albedo.g;
            mat.albedo[2]=m.albedo.b;
            mat.params[0]=1; mat.params[1]=1/std::max(m.roughness,1.f);
            mat.params[2]=3;
            break;
        case SceneDescription::MaterialType::Lambertian:
        default:
            mat.albedo[0]=m.albedo.r; mat.albedo[1]=m.albedo.g;
            mat.albedo[2]=m.albedo.b;
            mat.params[0]=1; mat.params[1]=1/std::max(m.roughness,1.f);
            mat.params[2]=2;
            {
                float lum = 0.299f*mat.albedo[0] + 0.587f*mat.albedo[1] + 0.114f*mat.albedo[2];
                mat.precalc[0] = std::log(0.07f + 0.93f * lum);
            }
            break;
        }
        materials.push_back(mat);
    }
}

void buildLights(const SceneDescription& desc,
                 std::vector<AccelerationStructure::GpuLight>& gpuLights,
                 uint32_t maxLights)
{
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
        if (gpuLights.size()>=maxLights) break;
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
        if (gpuLights.size()>=maxLights) break;
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
        if (gpuLights.size()>=maxLights) break;
        AccelerationStructure::GpuLight l{};
        l.pos_type[0]=dl.dir.x; l.pos_type[1]=dl.dir.y;
        l.pos_type[2]=dl.dir.z; l.pos_type[3]=1;
        l.color_intensity[0]=dl.color.r;
        l.color_intensity[1]=dl.color.g;
        l.color_intensity[2]=dl.color.b;
        l.color_intensity[3]=dl.intensity;
        gpuLights.push_back(l);
    }
    while (gpuLights.size() < maxLights)
        gpuLights.push_back(AccelerationStructure::GpuLight{});
}

} // namespace

void SceneBuilder::build(const SceneDescription& desc,
                          AccelerationStructure& as)
{
    as.maxMaterials = desc.maxMaterials;
    as.maxLights    = desc.maxLights;
    as.maxInstances = desc.maxInstances;

    std::vector<AccelerationStructure::InstanceInfo> instances;
    std::vector<AccelerationStructure::MaterialGPU>  materials;
    buildInstances(desc, as, instances, materials);

    std::vector<AccelerationStructure::GpuLight> gpuLights;
    buildLights(desc, gpuLights, as.maxLights);

    as.setDiffuseStrength(desc.diffuseStrength);
    as.setSpecularStrength(desc.specularStrength);
    as.buildScene(instances, materials, gpuLights);
}
