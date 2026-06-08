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
    m_surface = m_window->createSurface(m_ctx->getInstance());

    // 4. Check present support for the compute queue
    auto& qf = const_cast<VulkanContext::QueueFamilyIndices&>(m_ctx->getQueueFamilies());
    auto physDev = m_ctx->getPhysicalDevice();
    bool presentSupported = physDev.getSurfaceSupportKHR(
        qf.compute.value(), *m_surface
    );
    if (!presentSupported) {
        // Try to find a different present family
        auto queueProps = physDev.getQueueFamilyProperties();
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueProps.size()); ++i) {
            if (physDev.getSurfaceSupportKHR(i, *m_surface)) {
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
        m_surface,
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
    // Phase 3: BK7 glass prism — two non-parallel faces forming a wedge.
    //
    // Camera at (0,0,-5) looking +Z.
    // Rays enter the entry face (Z≈2-3), travel through the glass interior,
    // and exit through the exit face (Z≈3-4). The two faces share an apex at
    // (0, 1.2, 3.0). Different wavelengths refract at different angles
    // (Cauchy dispersion) → rainbow projects onto the sky.
    //
    // Wedge angle ≈ 56° between face normals.
    // Both faces use BK7 glass with per-channel Cauchy coefficients.

    // Entry face — ~37° tilt, large, covers entire view
    AccelerationStructure::InstanceInfo entryFace;
    entryFace.mesh.vertices = {
        -8.0f, -4.0f,  2.0f,
         8.0f, -4.0f, 12.0f,
         0.0f,  8.0f,  2.0f,
    };
    entryFace.mesh.indices = {0, 1, 2};
    entryFace.customIndex  = 0;
    entryFace.materialID   = 0;

    // Dummy second instance (never hit); full prism needs a closed mesh volume
    AccelerationStructure::InstanceInfo exitFace;
    exitFace.mesh.vertices = {
        -0.01f, 0.0f,  100.0f,
         0.01f, 0.0f,  100.0f,
         0.0f,   0.01f, 100.0f,
    };
    exitFace.mesh.indices = {0, 1, 2};
    exitFace.customIndex  = 1;
    exitFace.materialID   = 0;   // same BK7 material

    std::vector<AccelerationStructure::MaterialGPU> materials(1);
    // BK7 glass: n(λ) = A + B/λ²  (λ in μm)
    // Coefficients from design doc §10.A:
    //   A_r=1.513 A_g=1.519 A_b=1.528
    //   B_r=0.0045 B_g=0.0045 B_b=0.0045
    auto& bk7 = materials[0];
    bk7.cauchyA[0] = 1.513f; bk7.cauchyA[1] = 1.519f; bk7.cauchyA[2] = 1.528f; bk7.cauchyA[3] = 0.0f;
    bk7.cauchyB[0] = 0.0045f; bk7.cauchyB[1] = 0.0045f; bk7.cauchyB[2] = 0.0045f; bk7.cauchyB[3] = 0.0f;
    bk7.params[0] = 1.517f;  // base IOR (sodium D-line)
    bk7.params[1] = 0.0f;     // roughness
    bk7.params[2] = 0.0f;     // DIELECTRIC
    bk7.params[3] = 0.0f;

    m_as->buildTwoInstance(entryFace, exitFace, materials);
    std::cout << "  Built BLASx2 + TLAS: BK7 prism wedge for adaptive spectral tracing." << std::endl;
}

void Application::run() {
    std::cout << "Entering render loop..." << std::endl;
    while (m_window->isOpen()) {
        if (!m_window->pollEvents()) break;

        m_renderer->renderFrame(*m_as, *m_pipeline);

        // Save first frame as PPM
        if (m_firstFrame) {
            m_renderer->saveOutputPNG("output.png");
            m_firstFrame = false;
        }
    }
}
