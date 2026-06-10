cmake_minimum_required(VERSION 4.3.1)
project(VKRT
    VERSION 0.1.0
    DESCRIPTION "Adaptive Spectral Tracing Renderer"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Vulkan SDK
# -----------------------------------------------------------------------------
set(VULKAN_SDK "C:/VulkanSDK/1.4.341.1")
set(CMAKE_PREFIX_PATH "${VULKAN_SDK}/cmake" ${CMAKE_PREFIX_PATH})

# Vulkan (via CMake built-in FindVulkan)
find_package(Vulkan REQUIRED)

# SDL3
find_package(SDL3 REQUIRED)

# -----------------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------------
set(SDK_INCLUDE "${VULKAN_SDK}/Include")
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third-party")

# GLM (header-only)
set(GLM_INCLUDE "${SDK_INCLUDE}/glm")

# Slang (header for now, future shader compilation)
set(SLANG_INCLUDE "${SDK_INCLUDE}/slang")

# -----------------------------------------------------------------------------
# Third-party sources
# -----------------------------------------------------------------------------
# tinyobjloader
add_library(tinyobjloader STATIC
    "${THIRD_PARTY_DIR}/tinyobjloader-1.0.6/tiny_obj_loader.cc"
)
target_include_directories(tinyobjloader PUBLIC
    "${THIRD_PARTY_DIR}/tinyobjloader-1.0.6"
)
target_compile_definitions(tinyobjloader PRIVATE TINYOBJLOADER_IMPLEMENTATION)

# tinyxml2
add_library(tinyxml2 STATIC
    "${THIRD_PARTY_DIR}/tinyxml2-11.0.0/tinyxml2.cpp"
)
target_include_directories(tinyxml2 PUBLIC
    "${THIRD_PARTY_DIR}/tinyxml2-11.0.0"
)

# stb_image (header-only, provide as interface library)
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE
    "${THIRD_PARTY_DIR}/stb"
)

# -----------------------------------------------------------------------------
# Shader compilation (glslc)
# -----------------------------------------------------------------------------
set(GLSLC "${VULKAN_SDK}/Bin/glslc.exe")
set(SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")

file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

function(compile_shader TARGET_NAME SRC_FILE OUTPUT_NAME)
    set(OUTPUT_FILE "${SHADER_OUT_DIR}/${OUTPUT_NAME}")
    add_custom_command(
        OUTPUT "${OUTPUT_FILE}"
        COMMAND "${GLSLC}" --target-env=vulkan1.4 -O
                -o "${OUTPUT_FILE}"
                "${SHADER_SRC_DIR}/${SRC_FILE}"
        DEPENDS "${SHADER_SRC_DIR}/${SRC_FILE}"
        COMMENT "Compiling shader: ${SRC_FILE} -> ${OUTPUT_NAME}"
    )
    add_custom_target(${TARGET_NAME} DEPENDS "${OUTPUT_FILE}")
    set(SHADER_OUTPUT_${TARGET_NAME} "${OUTPUT_FILE}" PARENT_SCOPE)
endfunction()

compile_shader(raytrace_shader raytrace.comp raytrace.comp.spv)
compile_shader(denoise_shader denoise.comp denoise.comp.spv)

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
add_executable(vkrt
    src/app/main.cpp
    src/app/Application.cpp
    src/app/SceneBuilder.cpp
    src/core/Window.cpp
    src/core/VulkanContext.cpp
    src/core/GPUBuffer.cpp
    src/core/Spectral.h
    src/ray/AccelerationStructure.cpp
    src/ray/RayTracingPipeline.cpp
    src/ray/DescriptorManager.cpp
    src/ray/EnvMap.cpp
    src/ray/GeometryBuffer.cpp
    src/ray/AccelBuilder.cpp
    src/render/CrossFrameAccum.cpp
    src/render/Denoiser.cpp
    src/render/FrameCapture.cpp
    src/render/FrameRecorder.cpp
    src/render/OutputImage.cpp
    src/render/Renderer.cpp
    src/render/ScreenshotCapture.cpp
    src/render/SwapchainManager.cpp
    src/scene/SceneConfig.cpp
    src/scene/SceneXmlParser.cpp
    src/scene/ObjLoader.cpp
)

target_include_directories(vkrt PRIVATE
    src
    ${Vulkan_INCLUDE_DIRS}
    ${GLM_INCLUDE}
    ${SLANG_INCLUDE}
)

target_link_libraries(vkrt PRIVATE
    Vulkan::Vulkan
    SDL3::SDL3-static
    tinyobjloader
    tinyxml2
    stb_image
)

# Ensure shaders are compiled before the executable
add_dependencies(vkrt raytrace_shader denoise_shader)

# Copy shader SPIR-V to runtime output directory
add_custom_command(TARGET vkrt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SHADER_OUT_DIR}/raytrace.comp.spv"
        "$<TARGET_FILE_DIR:vkrt>/shaders/raytrace.comp.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SHADER_OUT_DIR}/denoise.comp.spv"
        "$<TARGET_FILE_DIR:vkrt>/shaders/denoise.comp.spv"
    COMMENT "Copying shaders to output directory"
)

# -----------------------------------------------------------------------------
# Platform-specific settings
# -----------------------------------------------------------------------------
if(WIN32)
    # Use static runtime for SDL3 compatibility
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
