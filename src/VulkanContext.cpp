#include "VulkanContext.h"

#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Utility: debug messenger callback
// -----------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void*                                       /*pUserData*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << pCallbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

// -----------------------------------------------------------------------------
// Device extensions required for ray query pipeline
// -----------------------------------------------------------------------------
static const std::vector<const char*> s_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
};

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------
VulkanContext::VulkanContext(const std::vector<const char*>& instanceExtensions) {
    createInstance(instanceExtensions);
    pickPhysicalDevice();
    createDevice();
}

VulkanContext::~VulkanContext() {
    // vk::raii objects auto-cleanup in reverse declaration order.
    // Device before instance is handled automatically.
}

// -----------------------------------------------------------------------------
// Create Vulkan Instance
// -----------------------------------------------------------------------------
void VulkanContext::createInstance(const std::vector<const char*>& surfaceExtensions) {
    // Application info
    vk::ApplicationInfo appInfo(
        "VKRT", VK_MAKE_VERSION(0, 1, 0),
        "VKRT Engine", VK_MAKE_VERSION(0, 1, 0),
        VK_API_VERSION_1_4
    );

    // Collect all required extensions
    std::vector<const char*> extensions = surfaceExtensions;

    if constexpr (s_enableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Check extension support
    auto availableExtensions = m_context.enumerateInstanceExtensionProperties();
    std::set<std::string> availableSet;
    for (const auto& ext : availableExtensions) {
        availableSet.insert(ext.extensionName);
    }
    for (const auto* ext : extensions) {
        if (!availableSet.count(ext)) {
            std::cerr << "Warning: instance extension not available: " << ext << std::endl;
        }
    }

    vk::InstanceCreateInfo createInfo({}, &appInfo);

    // Validation layers
    if constexpr (s_enableValidation) {
        createInfo.enabledLayerCount   = 1;
        createInfo.ppEnabledLayerNames = &s_validationLayer;
    }

    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    m_instance = vk::raii::Instance(m_context, createInfo);

    // Debug messenger (only if validation is on)
    if constexpr (s_enableValidation) {
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo(
            {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debugCallback
        );
        m_debugMessenger = vk::raii::DebugUtilsMessengerEXT(m_instance, debugInfo);
    }
}

// -----------------------------------------------------------------------------
// Pick Physical Device
// -----------------------------------------------------------------------------
void VulkanContext::pickPhysicalDevice() {
    auto devices = m_instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found.");
    }

    for (auto& device : devices) {
        auto props       = device.getProperties();
        auto features    = device.getFeatures();
        auto memProps    = device.getMemoryProperties();

        // Prefer discrete GPU
        bool isDiscrete = (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu);

        // Check ray query support
        auto rtFeatures = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceRayQueryFeaturesKHR
        >();
        bool hasRayQuery = rtFeatures.get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;

        // Check acceleration structure support
        auto asFeatures = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        >();
        bool hasAccelStruct = asFeatures.get<
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        >().accelerationStructure;

        if (isDiscrete && hasRayQuery && hasAccelStruct) {
            m_physicalDevice = std::move(device);

            // Store ray tracing properties
            auto rtProps = m_physicalDevice.getProperties2<
                vk::PhysicalDeviceProperties2,
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR
            >();
            m_rtProperties = rtProps.get<
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR
            >();
            return;
        }
    }

    throw std::runtime_error("No GPU with ray query + acceleration structure support found.");
}

// -----------------------------------------------------------------------------
// Create Logical Device & Queues
// -----------------------------------------------------------------------------
void VulkanContext::createDevice() {
    // Find queue families
    auto queueProps = m_physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i) {
        if (queueProps[i].queueFlags & vk::QueueFlagBits::eCompute) {
            m_queueFamilies.compute = i;
        }
        // Present support is checked later with surface; for now pick a separate family if
        // possible, or fall back to compute family.
        if (queueProps[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            m_queueFamilies.present = i;
        }
    }

    // Present support requires checking against the surface.
    // For Phase 1, we set present = compute family as fallback.
    if (!m_queueFamilies.present.has_value()) {
        m_queueFamilies.present = m_queueFamilies.compute;
    }

    if (!m_queueFamilies.isComplete()) {
        throw std::runtime_error("Failed to find suitable queue families.");
    }

    // Create queues (use unique families set)
    std::set<uint32_t> uniqueFamilies = {
        *m_queueFamilies.compute,
        *m_queueFamilies.present
    };

    float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    for (uint32_t family : uniqueFamilies) {
        queueInfos.push_back({{}, family, 1, &queuePriority});
    }

    // Enable features
    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeature(true);
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeature;
    accelFeature.setAccelerationStructure(true);
    accelFeature.setPNext(&rayQueryFeature);
    vk::PhysicalDeviceFeatures2 features2;
    features2.setPNext(&accelFeature);

    // Create device
    vk::DeviceCreateInfo deviceInfo(
        {},
        static_cast<uint32_t>(queueInfos.size()),
        queueInfos.data()
    );
    deviceInfo.setPNext(&features2);
    deviceInfo.enabledExtensionCount   = static_cast<uint32_t>(s_deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = s_deviceExtensions.data();

    m_device = vk::raii::Device(m_physicalDevice, deviceInfo);

    // Retrieve queues
    m_computeQueue = m_device.getQueue(*m_queueFamilies.compute, 0);
    m_presentQueue = m_device.getQueue(*m_queueFamilies.present, 0);
}
