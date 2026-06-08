#include "Window.h"

#include <stdexcept>

Window::Window(int width, int height, const std::string& title) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }

    m_window = SDL_CreateWindow(
        title.c_str(), width, height,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
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
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) {
            m_open = false;
            return false;
        }
        if (event.type == SDL_EVENT_KEY_DOWN) {
            if (event.key.key == SDLK_ESCAPE) {
                m_open = false;
                return false;
            }
        }
    }
    return m_open;
}

std::pair<int, int> Window::getFramebufferSize() const {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(m_window, &w, &h);
    return {w, h};
}
