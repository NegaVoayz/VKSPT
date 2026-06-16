#pragma once

#include "core/VulkanContext.h"

#include <vulkan/vulkan_raii.hpp>

#include <vector>

/// Select compute and present queue families, return DeviceQueueCreateInfo vector.
std::vector<vk::DeviceQueueCreateInfo> SetupDeviceQueues(
    const vk::raii::PhysicalDevice& physDevice,
    VulkanContext::QueueFamilyIndices& families);
