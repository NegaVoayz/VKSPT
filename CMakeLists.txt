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
# Shader compilation (Slang)
# -----------------------------------------------------------------------------
set(SLANGC "${VULKAN_SDK}/Bin/slangc.exe")
set(SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")

file(MAKE_DIRECTORY "${SHADER_OUT_DIR}")

set(SHADER_HEADERS
    "${SHADER_SRC_DIR}/common/types.slang"
    "${SHADER_SRC_DIR}/common/cmf_data.slang"
    "${SHADER_SRC_DIR}/common/bindings.slang"
    "${SHADER_SRC_DIR}/math/spectral_math.slang"
    "${SHADER_SRC_DIR}/math/color.slang"
    "${SHADER_SRC_DIR}/math/absorption.slang"
    "${SHADER_SRC_DIR}/math/legendre_decay.slang"
    "${SHADER_SRC_DIR}/math/sampling.slang"
    "${SHADER_SRC_DIR}/geometry/dispersion.slang"
    "${SHADER_SRC_DIR}/geometry/normals.slang"
    "${SHADER_SRC_DIR}/materials/metal.slang"
    "${SHADER_SRC_DIR}/materials/checkerboard.slang"
    "${SHADER_SRC_DIR}/materials/lambertian.slang"
    "${SHADER_SRC_DIR}/materials/dielectric.slang"
    "${SHADER_SRC_DIR}/materials/sky_handler.slang"
    "${SHADER_SRC_DIR}/materials/direct_lighting.slang"
)

function(compile_shader_slang TARGET_NAME SRC_FILE OUTPUT_NAME)
    cmake_parse_arguments(ARG "" "" "SLANGC_FLAGS" ${ARGN})
    set(OUTPUT_FILE "${SHADER_OUT_DIR}/${OUTPUT_NAME}")
    add_custom_command(
        OUTPUT "${OUTPUT_FILE}"
        COMMAND "${SLANGC}"
                "${SHADER_SRC_DIR}/${SRC_FILE}"
                -target spirv
                -emit-spirv-directly
                -entry main
                -stage compute
                -I "${SHADER_SRC_DIR}"
                ${ARG_SLANGC_FLAGS}
                -o "${OUTPUT_FILE}"
        DEPENDS "${SHADER_SRC_DIR}/${SRC_FILE}"
                ${SHADER_HEADERS}
        COMMENT "Compiling Slang shader: ${SRC_FILE} -> ${OUTPUT_NAME}"
    )
    add_custom_target(${TARGET_NAME} DEPENDS "${OUTPUT_FILE}")
    set(SHADER_OUTPUT_${TARGET_NAME} "${OUTPUT_FILE}" PARENT_SCOPE)
endfunction()

compile_shader_slang(denoise_shader denoise.slang denoise.comp.spv)

# RT Pipeline shader (multi-entry-point RT stages)
set(RT_PIPELINE_SRC "${SHADER_SRC_DIR}/raytrace_pipeline.slang")
set(RT_PIPELINE_OUT "${SHADER_OUT_DIR}/raytrace_pipeline.spv")
add_custom_command(
    OUTPUT "${RT_PIPELINE_OUT}"
    COMMAND "${SLANGC}"
            "${RT_PIPELINE_SRC}"
            -target spirv
            -emit-spirv-directly
            -I "${SHADER_SRC_DIR}"
            -entry RayGenMain -stage raygeneration
            -entry ClosestHitMain -stage closesthit
            -entry MissMain -stage miss
            -capability spvRayTracingKHR
            -o "${RT_PIPELINE_OUT}"
    DEPENDS "${RT_PIPELINE_SRC}"
            ${SHADER_HEADERS}
    COMMENT "Compiling Slang shader: raytrace_pipeline.slang -> raytrace_pipeline.spv"
)
add_custom_target(raytrace_pipeline_shader DEPENDS "${RT_PIPELINE_OUT}")

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

# Enable dynamic dispatch (required for VK_KHR_ray_tracing_pipeline functions)
target_compile_definitions(vkrt PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

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
add_dependencies(vkrt denoise_shader raytrace_pipeline_shader)

# Copy shader SPIR-V to runtime output directory
add_custom_command(TARGET vkrt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SHADER_OUT_DIR}/denoise.comp.spv"
        "$<TARGET_FILE_DIR:vkrt>/shaders/denoise.comp.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SHADER_OUT_DIR}/raytrace_pipeline.spv"
        "$<TARGET_FILE_DIR:vkrt>/shaders/raytrace_pipeline.spv"
    COMMENT "Copying shaders to output directory"
)

# -----------------------------------------------------------------------------
# Platform-specific settings
# -----------------------------------------------------------------------------
if(WIN32)
    # Use static runtime for SDL3 compatibility
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
