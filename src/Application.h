#pragma once

#include "AccelerationStructure.h"
#include "RayTracingPipeline.h"
#include "Renderer.h"
#include "VulkanContext.h"
#include "Window.h"

#include <memory>
#include <string>

/// Top-level application: owns all subsystems and runs the main loop.
class Application {
public:
    Application(int windowWidth, int windowHeight, const std::string& title);
    ~Application();

    // Non-copyable, non-movable
    Application(const Application&)            = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&)                 = delete;
    Application& operator=(Application&&)      = delete;

    /// Run the render loop until the window is closed.
    void run();

private:
    void initScene();

    std::unique_ptr<Window>              m_window;
    std::unique_ptr<VulkanContext>       m_ctx;
    std::unique_ptr<AccelerationStructure> m_as;
    std::unique_ptr<RayTracingPipeline>  m_pipeline;
    std::unique_ptr<Renderer>            m_renderer;

    uint32_t m_width;
    uint32_t m_height;
    bool     m_firstFrame = true;
};
