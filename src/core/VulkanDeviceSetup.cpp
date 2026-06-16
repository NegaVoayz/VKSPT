#include "core/VulkanDeviceSetup.h"

#include <set>
#include <stdexcept>

std::vector<vk::DeviceQueueCreateInfo> SetupDeviceQueues(
    const vk::raii::PhysicalDevice& physDevice,
    VulkanContext::QueueFamilyIndices& families)
{
    auto queueProps = physDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i) {
        if (queueProps[i].queueFlags & vk::QueueFlagBits::eCompute)
            families.compute = i;
        if (queueProps[i].queueFlags & vk::QueueFlagBits::eGraphics)
            families.present = i;
    }

    if (!families.present.has_value())
        families.present = families.compute;

    if (!families.IsComplete())
        throw std::runtime_error("Failed to find suitable queue families.");

    std::set<uint32_t> uniqueFamilies = {
        *families.compute,
        *families.present
    };

    float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    for (uint32_t family : uniqueFamilies)
        queueInfos.push_back({{}, family, 1, &queuePriority});

    return queueInfos;
}
