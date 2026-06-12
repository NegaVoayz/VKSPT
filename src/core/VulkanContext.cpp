#include "core/VulkanContext.h"
#include "core/Log.h"

#include <cstring>
#include <set>
#include <stdexcept>

// Debug messenger callback
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void*                                       /*pUserData*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        Log::error("[Vulkan] {}", pCallbackData->pMessage);
    }
    return VK_FALSE;
}

// Device extensions for ray tracing pipeline
static const std::vector<const char*> s_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
};

// Constructor
VulkanContext::VulkanContext(const std::vector<const char*>& instanceExtensions) {
    createInstance(instanceExtensions);
    pickPhysicalDevice();
    createDevice();
}

VulkanContext::~VulkanContext() {
    // vk::raii objects auto-cleanup in reverse declaration order.
    // Device before instance is handled automatically.
}

// Create Vulkan Instance
void VulkanContext::createInstance(const std::vector<const char*>& surfaceExtensions) {
    // Application info
    vk::ApplicationInfo appInfo(
        "VKRT", VK_MAKE_VERSION(0, 1, 0),
        "VKRT Engine", VK_MAKE_VERSION(0, 1, 0),
        VK_API_VERSION_1_4
    );

    std::vector<const char*> extensions = surfaceExtensions;

    if constexpr (s_enableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    auto availableExtensions = m_context.enumerateInstanceExtensionProperties();
    std::set<std::string> availableSet;
    for (const auto& ext : availableExtensions) {
        availableSet.insert(ext.extensionName);
    }
    for (const auto* ext : extensions) {
        if (!availableSet.count(ext)) {
            Log::warn("Instance extension not available: {}", ext);
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

    // Step 2: Init dynamic dispatcher with instance-level functions
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_instance);

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

        // Check ray tracing pipeline support
        auto rtFeatures = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR
        >();
        bool hasRTPipeline = rtFeatures.get<
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR
        >().rayTracingPipeline;

        // Check acceleration structure support
        auto asFeatures = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        >();
        bool hasAccelStruct = asFeatures.get<
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        >().accelerationStructure;

        if (isDiscrete && hasRTPipeline && hasAccelStruct) {
            m_physicalDevice = std::move(device);

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

    throw std::runtime_error("No GPU with ray tracing pipeline + acceleration structure support found.");
}

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

    if (!m_queueFamilies.present.has_value()) {
        m_queueFamilies.present = m_queueFamilies.compute;
    }

    if (!m_queueFamilies.isComplete()) {
        throw std::runtime_error("Failed to find suitable queue families.");
    }

    std::set<uint32_t> uniqueFamilies = {
        *m_queueFamilies.compute,
        *m_queueFamilies.present
    };

    float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    for (uint32_t family : uniqueFamilies) {
        queueInfos.push_back({{}, family, 1, &queuePriority});
    }

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature(true);
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeature;
    accelFeature.setAccelerationStructure(true);
    accelFeature.setPNext(&rtPipelineFeature);

    vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR bufferAddrFeature;
    bufferAddrFeature.setBufferDeviceAddress(true);
    bufferAddrFeature.setPNext(&accelFeature);

    vk::PhysicalDeviceFeatures2 features2;
    features2.setPNext(&bufferAddrFeature);

    vk::DeviceCreateInfo deviceInfo(
        {},
        static_cast<uint32_t>(queueInfos.size()),
        queueInfos.data()
    );
    deviceInfo.setPNext(&features2);
    deviceInfo.enabledExtensionCount   = static_cast<uint32_t>(s_deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = s_deviceExtensions.data();

    m_device = vk::raii::Device(m_physicalDevice, deviceInfo);

    // Init device-level function pointers for RT pipeline (vkCmdTraceRaysKHR etc.)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_device);

    m_computeQueue = m_device.getQueue(*m_queueFamilies.compute, 0);
    m_presentQueue = m_device.getQueue(*m_queueFamilies.present, 0);
}
