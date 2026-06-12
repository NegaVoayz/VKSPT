# Ray Query → Ray Tracing Pipeline 迁移问题记录

日期: 2026-06-12

## 概述

尝试将渲染器从 `VK_KHR_ray_query` 迁移到 `VK_KHR_ray_tracing_pipeline`。
C++ 端和 Shader 端基本完成（45 文件，+2319/-282 行），但运行时因工具链和 API 限制无法正常工作。

---

## 问题 1: Slang 编译器对 RT Pipeline Shader 的内置变量支持不完整

**状态**: 部分解决，采用混合方式绕过

**详情**:
- Slang 的 HLSL 模式（`.slang` 文件）中，ray generation shader 的 HLSL 内置变量可以正常使用：
  `DispatchRaysIndex()`, `DispatchRaysDimensions()`, `TraceRay()`, `RayDesc` 等。
- 但 closest-hit shader 中，HLSL 内置变量 `HitT()` 不被识别。
- GLSL 内置变量 `gl_HitTEXT`, `gl_WorldRayOriginEXT` 等在 `.slang` 文件中同样不被识别。
- 在 `.glsl` 文件中使用 `#version 460` + `#extension GL_EXT_ray_tracing : enable`，
  closest-hit 和 miss shader 可以通过 `gl_HitTEXT` 等 GLSL 内置变量正常编译。
- 但 `.glsl` 文件的 ray generation shader 在 slangc 编译时崩溃：
  "hit unreachable code: Unhandled case in emitSPIRVAsm"。

**绕过方案**: 
- Raygen 使用 `.slang` HLSL 模式
- Closest-hit / Miss / Shadow shader 使用 `.glsl` GLSL 模式
- Payload 声明使用标准 GLSL `layout(location = 0) inout RayPayload` 而非 `rayPayloadInOutEXT`

---

## 问题 2: Slang closest-hit 内调用 traceRayEXT 导致编译器崩溃

**状态**: 未解决，直接光照功能暂时禁用

**详情**:
- 当 GLSL closest-hit shader 中调用 `traceRayEXT()` 进行阴影光线追踪时，
  slangc 编译器内部崩溃：
  "Slang compilation aborted due to an exception of class Slang::InternalError: 
   hit unreachable code: Unhandled case in emitSPIRVAsm"
- 独立的 miss/closest-hit shader（不调用 traceRayEXT 的）可以正常编译。
- 这阻止了从 closest-hit 内部发起二级射线（阴影光线、反射、折射递归等）。

**绕过方案**:
- `chit_diffuse.glsl` 和 `chit_dielectric.glsl` 中的直接光照调用被禁用，
  替换为 `vec3(0.0)`。
- `shadow_chit.glsl` 简化为仅检查材质类型（电介质→通过，其他→阻挡），
  不进行递归追踪。
- 这意味着：无阴影、无焦散、无直接光照。仅保留环境光照（miss shader）。

---

## 问题 3: GLSL ray tracing shader 缺少部分内置变量

**状态**: 已绕过

**详情**:
- `gl_RayFlagsAcceptFirstHitAndEndSearchEXT` 在 Slang 编译的 `.glsl` shader 中不被识别。
- `gl_BaryCoordEXT` 同样不被识别。
- 改用硬编码数值和重心坐标近似值。

**绕过方案**:
- Ray flags 使用数值常量 `4` 替代 `gl_RayFlagsAcceptFirstHitAndEndSearchEXT`
- 重心坐标使用 `vec2(0.333, 0.333)`（三角形中心近似）

---

## 问题 4: Vulkan-Hpp 动态分发加载器链接问题

**状态**: 未解决，导致运行时崩溃

**详情**:
- `VK_KHR_ray_tracing_pipeline` 的扩展函数（`vkCmdTraceRaysKHR`,
  `vkGetRayTracingShaderGroupHandlesKHR`）无法通过静态分发链接。
  `vulkan-1.lib` 不导出这些带 KHR 后缀的符号。
- 切换到 `VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1` 动态分发后，
  需要 `VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE` 定义
  DispatchLoaderDynamic 实例。
- 尝试多种方式定义 storage：
  - 在 VulkanContext.cpp 中通过宏调用 → 无效（头文件已处理）
  - 在独立 VulkanDispatch.cpp 中 → 编译通过但运行时仍崩溃
- 运行时出现 Access Violation (0xC0000005)，无任何日志输出，
  崩溃发生在 VulkanContext 构造期间。
- 尝试 `VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1` 让 Context 自动加载
  vulkan-1.dll → 同样崩溃。

**未解决原因**:
- 无法定位具体崩溃位置（程序在第一批 Log::info() 输出前就崩溃了）
- 可能是动态分发加载器与 vulkan-1.dll 的初始化顺序问题
- 可能是 vk::raii::Context 与 DispatchLoaderDynamic 的交互问题

---

## 问题 5: Shader 和 C++ 之间的类型布局对齐

**状态**: 潜在问题

**详情**:
- RayBufferEntry 在 Slang/GLSL 和 C++ 之间的大小需要精确匹配。
  当前使用估算值 160 字节。
- RayPayload 在 GLSL 中使用 `std430` 布局，与 Slang 的布局可能有差异。
- 未在运行时验证。

---

## 已完成的 C++ 重构

以下模块已完整实现并编译通过：

| 模块 | 状态 |
|------|------|
| RayBufferManager（乒乓缓冲迭代弹跳循环） | ✅ |
| SBTBuilder（着色器绑定表构建） | ✅ |
| RayTracingPipeline（RT 管线创建） | ✅ |
| FrameRecorder（vkCmdTraceRaysKHR 弹跳循环） | ✅ |
| DescriptorManager（RT stage flags + 动态dispatch） | ✅ |
| AccelBuilder（per-instance SBT offsets） | ✅ |
| SceneBuilder（material → hit group 映射） | ✅ |
| ScreenshotCapture（适配 RT 管线） | ✅ |

---

## 总结

核心障碍是工具链不成熟（Slang 对 VK_KHR_ray_tracing_pipeline 支持不完整）
和 Vulkan-Hpp 动态分发的平台相关问题。
C++ 端架构设计是正确的，但运行时因这些底层问题无法工作。

建议等待 Slang 更新完善 RT pipeline 支持后再继续此迁移，
或考虑使用纯 GLSL + glslangValidator 编译所有 RT shader（需要完整的 GLSL 重写）。
