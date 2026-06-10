#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <string>
#include <vector>

/// Per-frame input state polled from SDL.
struct InputState {
    bool keyW = false, keyA = false, keyS = false, keyD = false;
    bool keyQ = false, keyE = false;  // keyQ=LSHIFT(down), keyE=SPACE(up)
    bool keyT = false;                // screenshot trigger
    bool mouseLeft  = false;
    float mouseDX = 0.0f, mouseDY = 0.0f;  // delta since last frame
    bool quitRequested = false;
};

/// RAII wrapper around SDL3 window + Vulkan surface.
/// Owns the SDL_Window and the VkSurfaceKHR throughout its lifetime.
class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    // Non-copyable, movable
    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&)                 = delete;
    Window& operator=(Window&&)      = delete;

    /// Returns the Vulkan instance extensions required by SDL3 for surface creation.
    static std::vector<const char*> getRequiredInstanceExtensions();

    /// Create the Vulkan surface for the given instance.
    vk::raii::SurfaceKHR createSurface(const vk::raii::Instance& instance) const;

    /// Poll SDL events. Returns false if quit requested.
    /// Fills m_input with current frame's input state.
    bool pollEvents();

    /// Returns false when the window should close.
    bool isOpen() const { return m_open; }

    /// Get the current frame's input state.
    const InputState& getInput() const { return m_input; }

    /// Get framebuffer size in pixels (may differ from window size on HiDPI).
    std::pair<int, int> getFramebufferSize() const;

    /// Toggle relative mouse mode (cursor hidden + locked for FPS look).
    void setRelativeMouse(bool enabled);

    SDL_Window* getHandle() const { return m_window; }

private:
    SDL_Window*  m_window  = nullptr;
    bool         m_open    = true;
    InputState   m_input;
    float        m_mouseAccumX = 0.0f, m_mouseAccumY = 0.0f;  // accumulator for relative mode
};
