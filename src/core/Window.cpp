#include "core/Window.h"

#include <stdexcept>

Window::Window(int width, int height, const std::string& title) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }

    m_window = SDL_CreateWindow(
        title.c_str(), width, height,
        SDL_WINDOW_VULKAN
    );
    if (!m_window) {
        SDL_Quit();
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }
}

Window::~Window() {
    if (m_window) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }
    SDL_Quit();
}

std::vector<const char*> Window::getRequiredInstanceExtensions() {
    // SDL3: use SDL_Vulkan_GetInstanceExtensions
    unsigned int count = 0;
    const char* const* exts = SDL_Vulkan_GetInstanceExtensions(&count);
    if (!exts) {
        throw std::runtime_error(
            std::string("SDL_Vulkan_GetInstanceExtensions failed: ") + SDL_GetError()
        );
    }
    return std::vector<const char*>(exts, exts + count);
}

vk::raii::SurfaceKHR Window::createSurface(const vk::raii::Instance& instance) const {
    VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(m_window, *instance, nullptr, &rawSurface)) {
        throw std::runtime_error(
            std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError()
        );
    }
    // Transfer ownership to the RAII wrapper
    return vk::raii::SurfaceKHR(instance, rawSurface);
}

bool Window::pollEvents() {
    // Reset per-frame deltas (key states persist across frames)
    m_input.mouseDX = m_mouseAccumX;
    m_input.mouseDY = m_mouseAccumY;
    m_mouseAccumX = 0.0f;
    m_mouseAccumY = 0.0f;
    m_input.quitRequested = false;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
        case SDL_EVENT_QUIT:
            m_open = false;
            m_input.quitRequested = true;
            return false;

        case SDL_EVENT_KEY_DOWN:
            if (event.key.key == SDLK_ESCAPE) {
                m_open = false;
                m_input.quitRequested = true;
                return false;
            }
            if (event.key.key == SDLK_W) m_input.keyW = true;
            if (event.key.key == SDLK_A) m_input.keyA = true;
            if (event.key.key == SDLK_S) m_input.keyS = true;
            if (event.key.key == SDLK_D) m_input.keyD = true;
            if (event.key.key == SDLK_LSHIFT) m_input.keyQ = true;
            if (event.key.key == SDLK_SPACE)  m_input.keyE = true;
            if (event.key.key == SDLK_T)      m_input.keyT = true;
            break;

        case SDL_EVENT_KEY_UP:
            if (event.key.key == SDLK_W) m_input.keyW = false;
            if (event.key.key == SDLK_A) m_input.keyA = false;
            if (event.key.key == SDLK_S) m_input.keyS = false;
            if (event.key.key == SDLK_D) m_input.keyD = false;
            if (event.key.key == SDLK_LSHIFT) m_input.keyQ = false;
            if (event.key.key == SDLK_SPACE)  m_input.keyE = false;
            if (event.key.key == SDLK_T)      m_input.keyT = false;
            break;

        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event.button.button == SDL_BUTTON_LEFT) {
                m_input.mouseLeft = true;
                setRelativeMouse(true);
            }
            break;

        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                m_input.mouseLeft = false;
                setRelativeMouse(false);
            }
            break;

        case SDL_EVENT_MOUSE_MOTION:
            // Only accumulate when left button is held
            if (m_input.mouseLeft) {
                m_mouseAccumX += event.motion.xrel;
                m_mouseAccumY += event.motion.yrel;
            }
            break;

        default:
            break;
        }
    }
    return m_open;
}

void Window::setRelativeMouse(bool enabled) {
    SDL_SetWindowRelativeMouseMode(m_window, enabled);
}

std::pair<int, int> Window::getFramebufferSize() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(m_window, &w, &h);
    return {w, h};
}
