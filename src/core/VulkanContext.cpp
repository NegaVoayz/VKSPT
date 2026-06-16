#include "core/VulkanContext.h"
#include "core/Log.h"

#include <cstring>
#include <set>
#include <stdexcept>

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

static const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

VulkanContext::VulkanContext(const std::vector<const char*>& instanceExtensions) {
    createInstance(instanceExtensions);
    pickPhysicalDevice();
    createDevice();
}

VulkanContext::~VulkanContext() = default;

void VulkanContext::createInstance(const std::vector<const char*>& surfaceExtensions) {
    vk::ApplicationInfo appInfo(
        "VKSPT", VK_MAKE_VERSION(0, 1, 0),
        "VKSPT Engine", VK_MAKE_VERSION(0, 1, 0),
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

    if constexpr (s_enableValidation) {
        createInfo.enabledLayerCount   = 1;
        createInfo.ppEnabledLayerNames = &s_validationLayer;
    }

    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    m_instance = vk::raii::Instance(m_context, createInfo);

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
        auto isDiscrete = (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu);

        auto rtFeatures = device.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR
        >();
        bool hasRTPipeline = rtFeatures.get<
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR
        >().rayTracingPipeline;

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

    throw std::runtime_error(
        "No GPU with ray tracing pipeline + acceleration structure support found.");
}

void VulkanContext::createDevice() {
    auto queueProps = m_physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i) {
        if (queueProps[i].queueFlags & vk::QueueFlagBits::eCompute) {
            m_queueFamilies.compute = i;
        }
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

    // pNext chain: all structs on this stack frame — driver reads through
    // the pointer chain during vkCreateDevice, so every link must outlive the call.
    vk::PhysicalDeviceVulkan11Features query11;
    vk::PhysicalDeviceVulkan12Features query12;
    query11.setPNext(&query12);
    {
        vk::PhysicalDeviceFeatures2 qf2;
        qf2.setPNext(&query11);
        static_cast<vk::PhysicalDevice>(m_physicalDevice).getFeatures2(&qf2);
    }
    bool hasUniform16bit = query11.uniformAndStorageBuffer16BitAccess;
    bool hasFloat16      = query12.shaderFloat16;
    Log::info("GPU features: uniformAndStorage16bit={}  float16={}",
        hasUniform16bit, hasFloat16);

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature;
    rtPipelineFeature.setRayTracingPipeline(true);
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeature;
    accelFeature.setAccelerationStructure(true);
    accelFeature.setPNext(&rtPipelineFeature);

    void* pNextChain = &accelFeature;

    vk::PhysicalDeviceVulkan11Features vk11features;
    if (hasUniform16bit) {
        vk11features.storageBuffer16BitAccess = VK_TRUE;
        vk11features.uniformAndStorageBuffer16BitAccess = VK_TRUE;
        vk11features.setPNext(pNextChain);
        pNextChain = &vk11features;
    }

    vk::PhysicalDeviceVulkan12Features vk12features;
    vk12features.bufferDeviceAddress = VK_TRUE;
    if (hasFloat16)
        vk12features.shaderFloat16 = VK_TRUE;
    vk12features.setPNext(pNextChain);

    vk::PhysicalDeviceFeatures2 features2;
    features2.setPNext(&vk12features);

    vk::DeviceCreateInfo deviceInfo(
        {},
        static_cast<uint32_t>(queueInfos.size()),
        queueInfos.data()
    );
    deviceInfo.setPNext(&features2);
    deviceInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();

    m_device = vk::raii::Device(m_physicalDevice, deviceInfo);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_device);

    m_computeQueue = m_device.getQueue(*m_queueFamilies.compute, 0);
    m_presentQueue = m_device.getQueue(*m_queueFamilies.present, 0);
}
