#include "Application.h"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        Application app(1280, 720, "VKRT — Spectral Ray Tracer (Phase 1)");
        app.run();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
