#include "ray/GeometryBuffer.h"

void GeometryBuffer::upload(
    const vk::raii::Device& dev,
    const vk::raii::PhysicalDevice& physDev,
    const std::vector<std::vector<float>>&    stagedVertices,
    const std::vector<std::vector<uint32_t>>& stagedIndices,
    const std::vector<std::vector<float>>&    stagedNormals,
    const std::vector<uint32_t>& matIDs,
    const std::vector<uint32_t>& smoothFlags)
{
    uint32_t mi = maxInstances;
    std::vector<uint32_t> rangeData(6 * mi, 0);
    auto* vOff = &rangeData[0 * mi];
    auto* vCnt = &rangeData[1 * mi];
    auto* iOff = &rangeData[2 * mi];
    auto* iCnt = &rangeData[3 * mi];
    auto* mIDs = &rangeData[4 * mi];
    auto* sFlg = &rangeData[5 * mi];

    std::vector<float>    allV;
    std::vector<uint32_t> allI;
    std::vector<float>    allN;

    size_t n = stagedVertices.size();
    for (size_t i = 0; i < n && i < maxInstances; ++i) {
        vOff[i] = static_cast<uint32_t>(allV.size());
        vCnt[i] = static_cast<uint32_t>(stagedVertices[i].size());
        iOff[i] = static_cast<uint32_t>(allI.size());
        iCnt[i] = static_cast<uint32_t>(stagedIndices[i].size());
        mIDs[i] = (i < matIDs.size()) ? matIDs[i] : 0;
        sFlg[i] = (i < smoothFlags.size()) ? smoothFlags[i] : 0;

        allV.insert(allV.end(), stagedVertices[i].begin(),
                    stagedVertices[i].end());
        allI.insert(allI.end(), stagedIndices[i].begin(),
                    stagedIndices[i].end());
        if (i < stagedNormals.size() && !stagedNormals[i].empty())
            allN.insert(allN.end(), stagedNormals[i].begin(),
                        stagedNormals[i].end());
    }

    auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress;

    if (!allV.empty())
        m_vertex = GPUBuffer::createStaging(dev, allV.data(),
            allV.size() * sizeof(float), usage, physDev);
    if (!allI.empty())
        m_index = GPUBuffer::createStaging(dev, allI.data(),
            allI.size() * sizeof(uint32_t), usage, physDev);
    if (!allN.empty())
        m_normal = GPUBuffer::createStaging(dev, allN.data(),
            allN.size() * sizeof(float), usage, physDev);
    else {
        float d[4] = {0, 1, 0, 0};
        m_normal = GPUBuffer::createStaging(dev, d, sizeof(d),
                                             usage, physDev);
    }
    m_range = GPUBuffer::createStaging(dev, rangeData.data(),
        rangeData.size() * sizeof(uint32_t), usage, physDev);
}
