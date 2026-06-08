#include "Application.h"

#include <iostream>
#include <stdexcept>

Application::Application(int windowWidth, int windowHeight, const std::string& title)
    : m_width(windowWidth)
    , m_height(windowHeight)
{
    // 1. Create window (SDL3)
    std::cout << "Creating window..." << std::endl;
    m_window = std::make_unique<Window>(m_width, m_height, title);

    // Update dimensions from actual framebuffer size
    auto [fbW, fbH] = m_window->getFramebufferSize();
    m_width  = static_cast<uint32_t>(fbW);
    m_height = static_cast<uint32_t>(fbH);

    // 2. Create Vulkan context
    std::cout << "Creating Vulkan context..." << std::endl;
    auto instanceExts = Window::getRequiredInstanceExtensions();
    m_ctx = std::make_unique<VulkanContext>(instanceExts);

    // 3. Create surface
    std::cout << "Creating surface..." << std::endl;
    auto surface = m_window->createSurface(m_ctx->getInstance());

    // 4. Check present support for the compute queue
    auto& qf = const_cast<VulkanContext::QueueFamilyIndices&>(m_ctx->getQueueFamilies());
    auto physDev = m_ctx->getPhysicalDevice();
    bool presentSupported = physDev.getSurfaceSupportKHR(
        qf.compute.value(), *surface
    );
    if (!presentSupported) {
        // Try to find a different present family
        auto queueProps = physDev.getQueueFamilyProperties();
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i) {
            if (physDev.getSurfaceSupportKHR(i, *surface)) {
                qf.present = i;
                presentSupported = true;
                break;
            }
        }
    }
    if (!presentSupported) {
        throw std::runtime_error("No queue family supports present.");
    }

    // 5. Create acceleration structure (hardcoded triangle)
    std::cout << "Building acceleration structures..." << std::endl;
    m_as = std::make_unique<AccelerationStructure>(
        m_ctx->getDevice(),
        m_ctx->getPhysicalDevice(),
        m_ctx->getComputeQueueFamily()
    );
    initScene();

    // 6. Create ray tracing pipeline
    std::cout << "Creating ray tracing pipeline..." << std::endl;
    m_pipeline = std::make_unique<RayTracingPipeline>(
        m_ctx->getDevice(), m_width, m_height
    );
    m_pipeline->createPipeline("shaders/raytrace.comp.spv");

    // 7. Create renderer
    std::cout << "Creating renderer..." << std::endl;
    Renderer::Config cfg{m_width, m_height};
    m_renderer = std::make_unique<Renderer>(
        m_ctx->getInstance(),
        m_ctx->getDevice(),
        m_ctx->getPhysicalDevice(),
        surface,
        m_ctx->getComputeQueueFamily(),
        m_ctx->getPresentQueueFamily(),
        cfg
    );

    std::cout << "Initialization complete!" << std::endl;
}

Application::~Application() {
    // Destroy in reverse dependency order (unique_ptr handles this)
}

void Application::initScene() {
    // Hardcoded triangle: a large triangle filling the view
    AccelerationStructure::MeshData mesh;
    mesh.vertices = {
        // Triangle 1: large triangle in the XY plane at Z=0, facing -Z
        -2.0f, -1.5f,  0.0f,
         2.0f, -1.5f,  0.0f,
         0.0f,  2.0f,  0.0f,
    };
    mesh.indices = {0, 1, 2};

    m_as->build(mesh);
    std::cout << "  Built BLAS + TLAS with 1 triangle." << std::endl;
}

void Application::run() {
    std::cout << "Entering render loop..." << std::endl;
    while (m_window->isOpen()) {
        if (!m_window->pollEvents()) break;

        m_renderer->renderFrame(*m_as, *m_pipeline);

        // Save first frame as PPM
        if (m_firstFrame) {
            m_renderer->saveOutputPPM("output.ppm");
            m_firstFrame = false;
        }
    }
}
