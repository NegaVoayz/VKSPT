cmake_minimum_required(VERSION 4.3.1)
project(VKRT
    VERSION 0.1.0
    DESCRIPTION "Adaptive Spectral Tracing Renderer"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
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

function(compile_shader TARGET_NAME SRC_FILE OUTPUT_NAME)
    set(OUTPUT_FILE "${SHADER_OUT_DIR}/${OUTPUT_NAME}")
    add_custom_command(
        OUTPUT "${OUTPUT_FILE}"
        COMMAND "${SLANGC}"
                "${SHADER_SRC_DIR}/${SRC_FILE}"
                -target spirv
                -o "${OUTPUT_FILE}"
                -entry main
                -stage compute
                -I "${SHADER_SRC_DIR}"
        DEPENDS
            "${SHADER_SRC_DIR}/${SRC_FILE}"
            "${SHADER_SRC_DIR}/spectral_common.slang"
        COMMENT "Compiling shader (Slang): ${SRC_FILE} -> ${OUTPUT_NAME}"
    )
    add_custom_target(${TARGET_NAME} DEPENDS "${OUTPUT_FILE}")
    set(SHADER_OUTPUT_${TARGET_NAME} "${OUTPUT_FILE}" PARENT_SCOPE)
endfunction()

compile_shader(raytrace_shader raytrace.slang raytrace.slang.spv)

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
add_executable(vkrt
    src/main.cpp
    src/Application.cpp
    src/Window.cpp
    src/VulkanContext.cpp
    src/GPUBuffer.cpp
    src/AccelerationStructure.cpp
    src/RayTracingPipeline.cpp
    src/Renderer.cpp
    src/Spectral.h
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
add_dependencies(vkrt raytrace_shader)

# Copy shader SPIR-V to runtime output directory
add_custom_command(TARGET vkrt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SHADER_OUT_DIR}/raytrace.slang.spv"
        "$<TARGET_FILE_DIR:vkrt>/shaders/raytrace.slang.spv"
    COMMENT "Copying shader to output directory"
)

# -----------------------------------------------------------------------------
# Platform-specific settings
# -----------------------------------------------------------------------------
if(WIN32)
    # Use static runtime for SDL3 compatibility
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
