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

    // Phase 4.5: Sorted ray tracing pipeline (verified correct)
    m_renderer->initSortedPipeline(*m_pipeline);

    std::cout << "Initialization complete!" << std::endl;
}

Application::~Application() {
    // Destroy in reverse dependency order (unique_ptr handles this)
}

void Application::initScene() {
    // Phase 4: Diamond prism + reflective ground plane.
    // Camera at (0,0,-5) looking +Z, 90° horizontal FOV.

    std::vector<AccelerationStructure::InstanceInfo> instances;
    std::vector<AccelerationStructure::MaterialGPU> materials(2);

    // ---- Instance 0: BK7 triangular prism ----
    // Triangular cross-section in XZ, extruded in Y.
    // Entry face at Z=2: rectangle facing camera.
    // Far faces meet at apex Z=10 (face angle 26.6°, below BK7 41.2° critical).
    AccelerationStructure::InstanceInfo prism;
    prism.mesh.vertices = {
        // Near rectangle (Z=2)
        -5.0f, -4.0f, 2.0f,   // 0: A  bottom-left
         5.0f, -4.0f, 2.0f,   // 1: B  bottom-right
         5.0f,  5.0f, 2.0f,   // 2: C  top-right
        -5.0f,  5.0f, 2.0f,   // 3: D  top-left
        // Apex edge (Z=6, depth=4, face angle 51° → incidence 39° < 41° critical)
         0.0f, -4.0f, 6.0f,   // 4: E  apex bottom
         0.0f,  5.0f, 6.0f,   // 5: F  apex top
    };
    // 8 triangles: entry, 2 slanted far faces, top, bottom (no exit cap — V-shape)
    prism.mesh.indices = {
        // Entry face (Z=2, outward -Z): D,C,B + D,B,A
        3, 2, 1,  3, 1, 0,
        // Top cap (Y=5, outward +Y): D,F,C  (C is (5,5,2), F is (0,5,10))
        3, 5, 2,
        // Bottom cap (Y=-4, outward -Y): A,B,E
        0, 1, 4,
        // Far-right face: B,C,F,E (B-C and F-E edges)
        1, 2, 5,  1, 5, 4,
        // Far-left face: A,E,F,D  (A-D and E-F edges)
        0, 4, 5,  0, 5, 3,
    };
    prism.customIndex = 0;
    prism.materialID  = 0;
    instances.push_back(prism);

    // ---- Instance 1: Ground plane for reflection test ----
    AccelerationStructure::InstanceInfo ground;
    ground.mesh.vertices = {
        -20.0f, -5.0f, -5.0f,
         20.0f, -5.0f, -5.0f,
         20.0f, -5.0f, 30.0f,
        -20.0f, -5.0f, 30.0f,
    };
    ground.mesh.indices = {0, 1, 2, 0, 2, 3};  // upward normal
    ground.customIndex = 1;
    ground.materialID  = 1;  // white Lambertian
    instances.push_back(ground);

    // ---- Material 0: BK7 Glass ----
    auto& glass = materials[0];
    glass.cauchyA[0] = 1.513f; glass.cauchyA[1] = 1.519f; glass.cauchyA[2] = 1.528f; glass.cauchyA[3] = 0.0f;
    glass.cauchyB[0] = 0.0045f; glass.cauchyB[1] = 0.0045f; glass.cauchyB[2] = 0.0045f; glass.cauchyB[3] = 0.0f;
    glass.params[0] = 1.517f; glass.params[2] = 0.0f;  // DIELECTRIC

    // ---- Material 1: White Lambertian ground ----
    auto& whiteGround = materials[1];
    whiteGround.albedo[0] = 0.85f; whiteGround.albedo[1] = 0.85f; whiteGround.albedo[2] = 0.85f;
    whiteGround.params[2] = 2.0f;  // LAMBERTIAN

    // ---- Flat white light SPD ----
    AccelerationStructure::GpuLight skyLight;
    for (int i = 0; i < 16; ++i) skyLight.spd[i] = 1.0f;

    m_as->buildScene(instances, materials, skyLight);
    std::cout << "  Built scene: diamond prism + white ground plane." << std::endl;
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
