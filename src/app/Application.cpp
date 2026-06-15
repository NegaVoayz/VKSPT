#include "app/Application.h"
#include "core/Log.h"
#include "scene/SceneConfig.h"
#include "scene/SceneXmlParser.h"
#include "app/SceneBuilder.h"

#include <chrono>
#include <glm/gtc/constants.hpp>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;

void Camera::computeVectors(float fovTan, float aspect,
                            glm::vec3& camU, glm::vec3& camV, glm::vec3& camW) const {
    float cp = cos(pitch), sp = sin(pitch);
    float cy = cos(yaw),   sy = sin(yaw);
    glm::vec3 forward(sy * cp, -sp, cy * cp);
    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
    glm::vec3 up    = glm::cross(right, forward);

    camW = forward;
    camU = right * fovTan * aspect;
    camV = up * fovTan;
}

namespace {
    uint32_t setupPresentQueue(VulkanContext& ctx,
                               const vk::raii::PhysicalDevice& physDev,
                               const vk::raii::SurfaceKHR& surface)
    {
        auto& qf = const_cast<VulkanContext::QueueFamilyIndices&>(
            ctx.GetQueueFamilies());
        if (physDev.getSurfaceSupportKHR(qf.compute.value(), *surface))
            return *qf.compute;
        auto queueProps = physDev.getQueueFamilyProperties();
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i) {
            if (physDev.getSurfaceSupportKHR(i, *surface)) {
                qf.present = i;
                return i;
            }
        }
        throw std::runtime_error("No queue family supports present.");
    }
}

Application::Application(int windowWidth, int windowHeight, const std::string& title)
    : m_window(windowWidth, windowHeight, title)
    , m_width(windowWidth)
    , m_height(windowHeight)
    , m_ctx(Window::GetRequiredInstanceExtensions())
    , m_surface(m_window.CreateSurface(m_ctx.GetInstance()))
    , m_as(m_ctx.GetDevice(), m_ctx.GetPhysicalDevice(), m_ctx.GetComputeQueueFamily())
    , m_pipeline(m_ctx.GetDevice(), m_ctx.GetPhysicalDevice(),
                 m_ctx.GetComputeQueueFamily(), windowWidth, windowHeight)
    , m_renderer(m_ctx.GetInstance(), m_ctx.GetDevice(), m_ctx.GetPhysicalDevice(),
                 m_surface, m_ctx.GetComputeQueueFamily(),
                 setupPresentQueue(m_ctx, m_ctx.GetPhysicalDevice(), m_surface),
                 Renderer::Config{static_cast<uint32_t>(windowWidth),
                                  static_cast<uint32_t>(windowHeight)})
{
    Log::info("Creating window...");
    auto [fbW, fbH] = m_window.GetFramebufferSize();
    m_width  = static_cast<uint32_t>(fbW);
    m_height = static_cast<uint32_t>(fbH);

    Log::info("Building acceleration structures...");
    initScene();

    Log::info("Creating ray tracing pipeline...");
    m_pipeline.CreateDenoisePipeline("shaders/denoise.comp.spv");
    m_pipeline.CreateHashCountPipeline("shaders/hash_count.comp.spv");
    m_pipeline.CreateHashScanPipeline("shaders/hash_scan.comp.spv");
    m_pipeline.CreateHashScatterPipeline("shaders/hash_scatter.comp.spv");
    m_pipeline.CreateHashAggregatePipeline("shaders/hash_aggregate.comp.spv");
    m_pipeline.CreateHashGatherPipeline("shaders/hash_gather.comp.spv");
    m_pipeline.CreateStatsOverlayPipeline("shaders/stats_overlay.comp.spv");
    m_pipeline.CreateRTPipeline("shaders/raytrace_pipeline.spv");

    Log::info("Initialization complete!");
}

Application::~Application() = default;

void Application::initScene() {
    const std::string configPath = "../../assets/SceneConfig.xml";
    Log::info("Parsing scene config: {}", configPath);
    auto desc = ParseSceneXML(configPath);
    Log::info("  Camera: {}x{}, max depth: {}, objects: {}, point lights: {}, env map: {}",
              desc.cameraWidth, desc.cameraHeight, desc.maxDepth,
              desc.objects.size(), desc.pointLights.size(),
              desc.envMapDisplay ? "yes" : "no");
    m_outputName = desc.outputName;
    SceneBuilder().build(desc, m_as);
    Log::info("  Build complete: {} instances, {} materials",
              m_as.getInstanceCount(), m_as.getMaterialCount());
    if (desc.envMapDisplay)
        m_as.loadEnvMap("../../assets/envmap.jpg");
    else
        Log::info("  Environment map disabled by config");
}

void Application::handleCameraInput(float deltaTime) {
    const auto& inp = m_window.GetInput();

    m_camera.yaw   -= inp.mouseDX * m_camera.lookSpeed;
    m_camera.pitch += inp.mouseDY * m_camera.lookSpeed;
    const float maxPitch = glm::pi<float>() * 0.49f;
    m_camera.pitch = glm::clamp(m_camera.pitch, -maxPitch, maxPitch);

    float cy = cos(m_camera.yaw), sy = sin(m_camera.yaw);
    glm::vec3 forward(sy, 0.0f, cy);
    glm::vec3 right(-cy, 0.0f, sy);

    float speed = m_camera.moveSpeed * deltaTime;
    if (inp.keyW) m_camera.position += forward * speed;
    if (inp.keyS) m_camera.position -= forward * speed;
    if (inp.keyD) m_camera.position += right * speed;
    if (inp.keyA) m_camera.position -= right * speed;
    if (inp.keyE) m_camera.position.y += speed;
    if (inp.keyQ) m_camera.position.y -= speed;
}

CameraParams Application::buildCameraParams(float aspect, float fovTan) {
    CameraParams cam;
    cam.origin[0] = m_camera.position.x;
    cam.origin[1] = m_camera.position.y;
    cam.origin[2] = m_camera.position.z;
    glm::vec3 cu, cv, cw;
    m_camera.computeVectors(fovTan, aspect, cu, cv, cw);
    cam.camU[0] = cu.x; cam.camU[1] = cu.y; cam.camU[2] = cu.z;
    cam.camV[0] = cv.x; cam.camV[1] = cv.y; cam.camV[2] = cv.z;
    cam.camW[0] = cw.x; cam.camW[1] = cw.y; cam.camW[2] = cw.z;
    return cam;
}

void Application::run() {
    Log::info("Entering render loop...");
    Log::info("  WASD = move, SPACE/SHIFT = up/down, Left Mouse = look");

    auto lastTime = Clock::now();
    float avgDt = 0.0f;

    while (m_window.IsOpen()) {
        if (!m_window.PollEvents()) break;

        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        if (dt > 0.1f) dt = 0.1f;

        if (avgDt == 0.0f)
            avgDt = dt;
        else
            avgDt = 0.9f * avgDt + 0.1f * dt;

        handleCameraInput(dt);

        float aspect = float(m_width) / float(m_height);
        float fovTan = 0.57735f;  // 60° horizontal FOV (tan 30°)
        CameraParams cam = buildCameraParams(aspect, fovTan);

        if (m_window.GetInput().keyF3 && !m_f3WasDown) m_showStats = !m_showStats;
        m_f3WasDown = m_window.GetInput().keyF3;

        float fps = avgDt > 0.0f ? 1.0f / avgDt : 0.0f;
        m_renderer.RenderFrame(m_as, m_pipeline, cam, m_showStats, fps);

        if (m_window.GetInput().keyT) {
            float capFovTan = 0.57735f;
            float capAspect = 2560.0f / 1440.0f;
            CameraParams capCam = buildCameraParams(capAspect, capFovTan);

            m_renderer.CaptureScreenshot(m_outputName, m_as, m_pipeline,
                                           capCam, 2560, 1440, 64);
        }
    }
}
