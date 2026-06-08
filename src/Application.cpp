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
    m_pipeline->createPipeline("shaders/raytrace.slang.spv");

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
    // Phase 2: Two-triangle wedge prism for chromatic dispersion.
    //
    // Face 0 (entry, Z≈0):   large triangle facing camera (-Z)
    // Face 1 (exit,  Z≈1.5): offset triangle, angled to form a wedge
    //
    // Light enters through face 0, refracts per wavelength (Cauchy IOR),
    // exits through face 1 at different angles → visible dispersion.

    AccelerationStructure::InstanceInfo entryFace;
    entryFace.mesh.vertices = {
        -1.0f, -1.0f,  0.0f,
         1.0f, -1.0f,  0.0f,
         0.0f,  1.5f,  0.0f,
    };
    entryFace.mesh.indices  = {0, 1, 2};
    entryFace.customIndex   = 0;    // shader: material index 0
    entryFace.materialID    = 0;

    AccelerationStructure::InstanceInfo exitFace;
    exitFace.mesh.vertices = {
        -0.8f, -1.0f,  1.5f,
         1.2f, -1.0f,  1.5f,
         0.0f,  1.5f,  2.0f,
    };
    exitFace.mesh.indices  = {0, 1, 2};
    exitFace.customIndex   = 1;    // shader: material index 1
    exitFace.materialID    = 1;

    // Material data: BK7 glass for both faces
    // Cauchy formula: n(λ) = A + B/λ²  (B given for λ in μm)
    std::vector<AccelerationStructure::MaterialGPU> materials(2);
    auto makeBK7 = [](AccelerationStructure::MaterialGPU& m) {
        // BK7 glass: A=(1.513, 1.519, 1.528), B=(0.0045, 0.0045, 0.0045)
        m.cauchyA[0] = 1.513f;  m.cauchyA[1] = 1.519f;  m.cauchyA[2] = 1.528f;  m.cauchyA[3] = 0.0f;
        m.cauchyB[0] = 0.0045f; m.cauchyB[1] = 0.0045f; m.cauchyB[2] = 0.0045f; m.cauchyB[3] = 0.0f;
        m.albedo[0]  = 1.0f;    m.albedo[1]  = 1.0f;    m.albedo[2]  = 1.0f;    m.albedo[3]  = 0.0f;
        m.params[0]  = 1.517f;  // base IOR
        m.params[1]  = 0.0f;    // roughness
        m.params[2]  = 0.0f;    // type = DIELECTRIC
        m.params[3]  = 0.0f;
    };
    makeBK7(materials[0]);
    makeBK7(materials[1]);

    m_as->buildTwoInstance(entryFace, exitFace, materials);
    std::cout << "  Built BLASx2 + TLAS with 2-instance prism wedge (BK7 glass)." << std::endl;
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
