#pragma once

#include <cstdio>
#include <print>

/// Minimal inline logger using C++23 std::print.
/// Zero state, zero allocation — just formatted console output with prefixes.
namespace Log {

template<typename... Args>
void info(std::format_string<Args...> fmt, Args&&... args) {
    std::print("[INFO] ");
    std::println(fmt, std::forward<Args>(args)...);
    fflush(stdout);
}

template<typename... Args>
void error(std::format_string<Args...> fmt, Args&&... args) {
    std::print(stderr, "[ERROR] ");
    std::println(stderr, fmt, std::forward<Args>(args)...);
}

template<typename... Args>
void warn(std::format_string<Args...> fmt, Args&&... args) {
    std::print(stderr, "[WARN] ");
    std::println(stderr, fmt, std::forward<Args>(args)...);
}

} // namespace Log
