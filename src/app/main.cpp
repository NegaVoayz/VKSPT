#include "app/Application.h"
#include "core/Log.h"

#include <stdexcept>

int main() {
    try {
        Application app(1280, 720, "VKRT — Spectral Ray Tracer (Phase 1)");
        app.run();
        return 0;
    }
    catch (const std::exception& e) {
        Log::error("Fatal error: {}", e.what());
        return 1;
    }
}
