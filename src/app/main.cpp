// Dynamic dispatch — must precede all Vulkan includes
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include "app/Application.h"
#include "core/Log.h"

#include <stdexcept>

// Provide storage for the default dynamic dispatcher (exactly once)
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

int main() {
    try {
        // Step 1: Load vkGetInstanceProcAddr
        vk::detail::DynamicLoader dl;
        auto getProc = dl.getProcAddress<PFN_vkGetInstanceProcAddr>(
            "vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(getProc);

        Application app(1280, 720, "VKRT — Spectral Ray Tracer (Phase 1)");
        app.run();
        return 0;
    }
    catch (const std::exception& e) {
        Log::error("Fatal error: {}", e.what());
        return 1;
    }
}
