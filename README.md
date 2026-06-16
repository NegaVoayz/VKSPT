# VKSPT — Adaptive Spectral Tracing Renderer

Vulkan 1.4 Ray Tracing 渲染器，连续光谱建模 (380–780nm)，光子映射间接光照，光谱色散，渐进式跨帧累积。

## 依赖

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| **Vulkan SDK** | 1.4.341.1+ | API, 编译器 (glslang), Slang shader 编译 |
| **SDL3** | 3.x | 窗口创建, 输入处理 |
| **GLM** | (随Vulkan SDK分发) | 线性代数 (头文件) |
| **Slang** | (随Vulkan SDK分发) | Shader 编译 (HLSL→SPIR-V) |
| **CMake** | 4.3+ | 构建系统 |
| **C++23 编译器** | MSVC 2022 17.4+, Clang 17+, GCC 14+ | 需要 `std::print` (`<print>`), `std::expected` |

以下依赖由 CMake `FetchContent` 自动下载，无需手动安装：

| 库 | 用途 |
|----|------|
| tinyxml2 | XML 场景文件解析 |
| tinyobjloader | OBJ 模型加载 |
| stb_image | 环境贴图 (HDR/PNG) 读写 |

## 编译

```bash
# 1. 安装 Vulkan SDK 1.4+ 和 SDL3
# 2. 修改 CMakeLists.txt 中的 VULKAN_SDK 路径指向你的安装位置

cmake -B build -S .
cmake --build build --config Release
```

编译产物位于 `build/Release/vkspt.exe`，Shader SPIR-V 自动复制到输出目录。

## 运行

```bash
cd build/Release
./vkspt.exe
```

程序默认加载 `../../assets/SceneConfig.xml`。首次启动窗口可能较暗——光子图需数十帧累积。

## 文档

| 文档 | 内容 | 什么时候看 |
|------|------|-----------|
| **`docs/UserGuide.md`** | 操作控制、场景 XML 配置格式、材质参数 | 写场景、改配置、不知道怎么操作 |
| **`docs/项目设计文档：自适应光谱追踪渲染器.md`** | 完整架构：渲染管线、光子映射、光谱系统、材质、后处理 | 理解整体设计、找模块关系 |
| **`docs/SpectralDispersionDesign.md`** | 色散向量 D/V 传播、光谱分裂算法、Legendre 重投影 | 改色散/分裂相关代码 |
| **`docs/SpectralColorConversion.md`** | RGB↔光谱双向转换：Wiener 基、CMF 加权反射率、桶 CMF 投影、GL 求积 | 改颜色/吸收/材质光谱相关代码 |
| **`docs/项目开发记录文档.md`** | 开发日志 (按时间) | 追溯某次改动的上下文 |
| **`docs/bug-report-half-struct-crash.md`** | 历史 Bug 记录 (pNext 链 use-after-free) | 参考用 |

## 代码结构

```
src/
  core/      基础: GPUBuffer, VulkanContext, Window, Log
  app/       应用层: main, Application, SceneBuilder
  scene/     场景: SceneConfig, SceneXmlParser, ObjLoader
  ray/       GPU 资源: AccelerationStructure, RayTracingPipeline, DescriptorManager
  render/    渲染: Renderer, FrameRecorder, PhotonRecorder, ScreenshotCapture, Denoiser
shaders/
  common/    共享: types, bindings, cmf_data, hash_utils
  math/      数学: spectral_math, color, legendre_decay, sampling
  geometry/  几何: normals, dispersion
  rt/        RT 管线: payload, material_handlers, sky, direct_light
  photon/    光子: photon_gather, photon_hit, photon_light, blend_photon, hash_*
```

## 许可证

Apache License 2.0
