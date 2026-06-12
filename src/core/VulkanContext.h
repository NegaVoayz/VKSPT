#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <optional>
#include <vector>

/// Holds global Vulkan state: instance, physical device, logical device, and queue info.
/// All Vulkan objects use vk::raii for automatic cleanup.
class VulkanContext {
public:
    struct QueueFamilyIndices {
        std::optional<uint32_t> compute;   // Must support compute + ray tracing pipeline
        std::optional<uint32_t> present;   // Must support present to surface

        bool isComplete() const { return compute.has_value() && present.has_value(); }
    };

    VulkanContext(const std::vector<const char*>& instanceExtensions);
    ~VulkanContext();

    // Non-copyable, non-movable (others hold references to internals)
    VulkanContext(const VulkanContext&)            = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&)                 = delete;
    VulkanContext& operator=(VulkanContext&&)      = delete;

    // --- Accessors ---
    const vk::raii::Instance&       getInstance()       const { return m_instance; }
    const vk::raii::PhysicalDevice& getPhysicalDevice()  const { return m_physicalDevice; }
    const vk::raii::Device&         getDevice()          const { return m_device; }
    const QueueFamilyIndices&       getQueueFamilies()   const { return m_queueFamilies; }

    vk::Queue getComputeQueue() const { return m_computeQueue; }
    vk::Queue getPresentQueue() const { return m_presentQueue; }

    uint32_t getComputeQueueFamily() const { return *m_queueFamilies.compute; }
    uint32_t getPresentQueueFamily() const { return *m_queueFamilies.present; }

    /// Query ray tracing properties for this physical device.
    const vk::PhysicalDeviceRayTracingPipelinePropertiesKHR& getRayTracingProperties() const {
        return m_rtProperties;
    }

private:
    void createInstance(const std::vector<const char*>& surfaceExtensions);
    void pickPhysicalDevice();
    void createDevice();

    static constexpr bool          s_enableValidation  = true;
    static constexpr const char*   s_validationLayer    = "VK_LAYER_KHRONOS_validation";

    vk::raii::Context                               m_context;
    vk::raii::Instance                              m_instance          = nullptr;
    vk::raii::DebugUtilsMessengerEXT                m_debugMessenger    = nullptr;
    vk::raii::PhysicalDevice                        m_physicalDevice    = nullptr;
    vk::raii::Device                                m_device            = nullptr;

    QueueFamilyIndices                              m_queueFamilies;
    vk::Queue                                       m_computeQueue      = nullptr;
    vk::Queue                                       m_presentQueue      = nullptr;

    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties;
};
