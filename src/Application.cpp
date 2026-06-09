#include "Application.h"
#include "SceneConfig.h"

#include <chrono>
#include <glm/gtc/constants.hpp>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// Camera
// =============================================================================

void Camera::computeVectors(float fovTan, float aspect,
                            glm::vec3& camU, glm::vec3& camV, glm::vec3& camW) const {
    // Forward direction from yaw/pitch
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

// =============================================================================
// Application
// =============================================================================

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
    m_pipeline->createDenoisePipeline("shaders/denoise.comp.spv");

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

    // Phase 5: Sorted pipeline shelved — not faster, not more precise.
    // Infrastructure retained: classify/process shaders, hit-cache PackedRay,
    // overflow spill-to-host, CPU sortBatchByAction.
    // m_renderer->initSortedPipeline(*m_pipeline);

    std::cout << "Initialization complete!" << std::endl;
}

Application::~Application() {
    // Destroy in reverse dependency order (unique_ptr handles this)
}

void Application::initScene() {
    // Phase 5: Load scene from XML config, OBJ meshes, and environment map.
    const std::string configPath = "../../assets/SceneConfig.xml";

    std::cout << "Parsing scene config: " << configPath << std::endl;
    auto desc = parseSceneXML(configPath);

    std::cout << "  Camera: " << desc.cameraWidth << "x" << desc.cameraHeight
              << ", max depth: " << desc.maxDepth
              << ", objects: " << desc.objects.size()
              << ", point lights: " << desc.pointLights.size()
              << ", env map: " << (desc.envMapDisplay ? "yes" : "no")
              << std::endl;

    // ---- Build instance list ----
    std::vector<AccelerationStructure::InstanceInfo> instances;
    std::vector<AccelerationStructure::MaterialGPU>  materials;

    for (size_t i = 0; i < desc.objects.size(); ++i) {
        const auto& obj = desc.objects[i];

        std::cout << "  Loading: " << obj.objFilename << std::endl;
        auto mesh = loadObjMesh(obj.objFilename);

        std::cout << "    vertices: " << mesh.vertices.size() / 3
                  << ", triangles: " << mesh.indices.size() / 3 << std::endl;

        // Compute transform matrix
        float xf[3][4];
        buildTransformMatrix(obj.scale, obj.rotation, obj.translation, xf);

        AccelerationStructure::InstanceInfo inst;
        inst.mesh        = std::move(mesh);
        inst.customIndex = static_cast<uint32_t>(i);
        inst.materialID  = static_cast<uint32_t>(i);  // one material per object for now
        inst.hasNormals  = obj.normalInterpolation && !inst.mesh.normals.empty();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                inst.transform[r][c] = xf[r][c];

        instances.push_back(std::move(inst));

        // ---- Build material from XML params ----
        AccelerationStructure::MaterialGPU mat{};
        if (obj.ior <= 0.0f) {
            // Metal (IOR=0 sentinel)
            mat.albedo[0] = obj.diffuse.r;
            mat.albedo[1] = obj.diffuse.g;
            mat.albedo[2] = obj.diffuse.b;
            mat.params[0] = 1.0f;
            mat.params[1] = 1.0f / std::max(obj.shininess, 1.0f);
            mat.params[2] = 1.0f;       // MATERIAL_METAL
        } else if (obj.ior > 1.01f) {
            // Dielectric (glass)
            // Cauchy dispersion: n(λ) = A + B/λ²  (λ in μm)
            // BK7: A≈1.5046, B≈0.00420 → dispersion across 380-780nm ≈ 0.008
            for (int c = 0; c < 3; ++c) {
                mat.cauchyA[c] = obj.ior;
                mat.cauchyB[c] = obj.dispersionB;   // per-material Cauchy B
                mat.absorpA[c] = obj.absorbA[c];
                mat.absorpB[c] = obj.absorbB[c];
            }
            mat.params[0] = obj.ior;    // base IOR
            mat.params[1] = 0.0f;       // roughness (smooth glass)
            mat.params[2] = 0.0f;       // MATERIAL_DIELECTRIC
        } else if (obj.objFilename.find("checkerboard") != std::string::npos) {
            // Checkerboard — procedural pattern in shader
            mat.albedo[0] = obj.diffuse.r;
            mat.albedo[1] = obj.diffuse.g;
            mat.albedo[2] = obj.diffuse.b;
            mat.params[0] = 1.0f;
            mat.params[1] = 1.0f / std::max(obj.shininess, 1.0f);
            mat.params[2] = 3.0f;       // MATERIAL_CHECKERBOARD
        } else {
            // Lambertian (diffuse)
            mat.albedo[0] = obj.diffuse.r;
            mat.albedo[1] = obj.diffuse.g;
            mat.albedo[2] = obj.diffuse.b;
            mat.params[0] = 1.0f;
            mat.params[1] = 1.0f / std::max(obj.shininess, 1.0f);  // roughness
            mat.params[2] = 2.0f;       // MATERIAL_LAMBERTIAN
        }
        materials.push_back(mat);
    }

    // ---- Light buffer (ambient + sky) ----
    AccelerationStructure::GpuLight skyLight{};
    float ambientScale = desc.ambient.strength;
    for (int i = 0; i < 16; ++i) {
        skyLight.spd[i] = std::max(ambientScale, 0.1f);
    }

    // Build the acceleration structures + upload materials/lights
    m_as->buildScene(instances, materials, skyLight);

    // Load environment map
    m_as->loadEnvMap("../../assets/envmap.jpg");

    std::cout << "  Scene built: " << instances.size() << " objects, "
              << materials.size() << " materials." << std::endl;
}

void Application::run() {
    std::cout << "Entering render loop..." << std::endl;
    std::cout << "  WASD = move (horizontal), SPACE/SHIFT = up/down, Left Mouse = look" << std::endl;

    auto lastTime = Clock::now();

    while (m_window->isOpen()) {
        if (!m_window->pollEvents()) break;

        // Delta time in seconds
        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;
        // Clamp to avoid huge jumps on first frame or after pause
        if (dt > 0.1f) dt = 0.1f;

        // ---- Update camera from input ----
        const auto& inp = m_window->getInput();

        // Mouse look
        m_camera.yaw   -= inp.mouseDX * m_camera.lookSpeed;
        m_camera.pitch += inp.mouseDY * m_camera.lookSpeed;
        const float maxPitch = glm::pi<float>() * 0.49f;
        m_camera.pitch = glm::clamp(m_camera.pitch, -maxPitch, maxPitch);

        // Horizontal movement (yaw only, ignore pitch)
        float cy = cos(m_camera.yaw), sy = sin(m_camera.yaw);
        glm::vec3 forward(sy, 0.0f, cy);   // horizontal projection
        glm::vec3 right(-cy, 0.0f, sy);   // strafe right

        float speed = m_camera.moveSpeed * dt;
        if (inp.keyW) m_camera.position += forward * speed;
        if (inp.keyS) m_camera.position -= forward * speed;
        if (inp.keyD) m_camera.position += right * speed;
        if (inp.keyA) m_camera.position -= right * speed;
        if (inp.keyE) m_camera.position.y += speed;   // SPACE
        if (inp.keyQ) m_camera.position.y -= speed;   // LSHIFT

        // Compute camera vectors for rendering
        float aspect = float(m_width) / float(m_height);
        float fovTan = 0.57735f;  // 60° horizontal FOV (tan 30°)
        Renderer::CameraParams cam;
        cam.origin[0] = m_camera.position.x;
        cam.origin[1] = m_camera.position.y;
        cam.origin[2] = m_camera.position.z;
        {
            glm::vec3 cu, cv, cw;
            m_camera.computeVectors(fovTan, aspect, cu, cv, cw);
            cam.camU[0] = cu.x; cam.camU[1] = cu.y; cam.camU[2] = cu.z;
            cam.camV[0] = cv.x; cam.camV[1] = cv.y; cam.camV[2] = cv.z;
            cam.camW[0] = cw.x; cam.camW[1] = cw.y; cam.camW[2] = cw.z;
        }

        m_renderer->renderFrame(*m_as, *m_pipeline, cam);

        // T key: capture high-resolution screenshot (2K, 64 frames)
        if (inp.keyT) {
            float capFovTan = 0.57735f;  // 60° horizontal FOV
            float capAspect = 2560.0f / 1440.0f;
            glm::vec3 cu, cv, cw;
            m_camera.computeVectors(capFovTan, capAspect, cu, cv, cw);
            Renderer::CameraParams capCam;
            capCam.origin[0] = m_camera.position.x;
            capCam.origin[1] = m_camera.position.y;
            capCam.origin[2] = m_camera.position.z;
            capCam.camU[0] = cu.x; capCam.camU[1] = cu.y; capCam.camU[2] = cu.z;
            capCam.camV[0] = cv.x; capCam.camV[1] = cv.y; capCam.camV[2] = cv.z;
            capCam.camW[0] = cw.x; capCam.camW[1] = cw.y; capCam.camW[2] = cw.z;

            m_renderer->captureScreenshot("screenshot.png", *m_as, *m_pipeline,
                                           capCam, 2560, 1440, 64);
        }
    }
}
