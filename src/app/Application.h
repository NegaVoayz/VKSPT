#pragma once

#include "ray/AccelerationStructure.h"
#include "ray/RayTracingPipeline.h"
#include "render/Renderer.h"
#include "core/VulkanContext.h"
#include "core/Window.h"

#include <glm/glm.hpp>

#include <memory>
#include <string>

/// First-person camera state.
struct Camera {
    glm::vec3 position{ 0.0f, 0.0f, -5.0f };
    float yaw   = 0.0f;     // radians, 0 = looking +Z
    float pitch = 0.0f;     // radians, 0 = horizon

    float moveSpeed = 3.0f;  // world units per second
    float lookSpeed = 0.002f;

    /// Compute the three camera basis vectors for push constants.
    /// camW = forward, camU = right × fovTan × aspect, camV = up × fovTan
    void computeVectors(float fovTan, float aspect,
                        glm::vec3& camU, glm::vec3& camV, glm::vec3& camW) const;
};

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
    vk::raii::SurfaceKHR                 m_surface = nullptr;
    std::unique_ptr<AccelerationStructure> m_as;
    std::unique_ptr<RayTracingPipeline>  m_pipeline;
    std::unique_ptr<Renderer>            m_renderer;

    Camera   m_camera;
    uint32_t m_width;
    uint32_t m_height;
    std::string m_outputName;
};
