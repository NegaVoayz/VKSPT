# VKSPT 用户手册

自适应光谱追踪渲染器 — 使用指南与场景配置文件参考。

---

## 一、编译与运行

### 1.1 编译

```bash
cmake -B build -S .
cmake --build build --config Release
```

需要: CMake 3.21+, Visual Studio 2022 (MSVC), Vulkan SDK 1.4+, SDL3.

### 1.2 运行

```bash
cd build/Release
./vkspt.exe
```

程序自动加载 `../../assets/SceneConfig.xml` (相对于exe路径). 窗口标题 `VKSPT — Spectral Ray Tracer`.

---

## 二、操作控制

| 按键 | 功能 |
|------|------|
| **W/A/S/D** | 前/左/后/右移动 |
| **空格** | 上升 |
| **左Shift** | 下降 |
| **鼠标移动** | 旋转视角 (俯仰限±88°) |
| **T** | 截图: 2560×1440, 64帧累积, 保存为PNG |
| **F3** | 切换统计叠加层 (光子数/射线数/帧时间) |

截图文件名由场景XML的 `<Camera outputname="..."/>` 决定, 保存到exe同级目录.

---

## 三、场景配置文件 (SceneConfig.xml)

### 3.1 基本结构

```xml
<?xml version="1.0" encoding="utf-8"?>
<SceneInfo>
    <!-- 全局设置 -->
    <Camera>...</Camera>
    <EnvironmentMap>...</EnvironmentMap>
    <ResourceLimits>...</ResourceLimits>

    <!-- 物体 (可多个) -->
    <ObjectName>
        <Args filename="..." display="1" normalinterpolation="0"/>
        <Scale x="1" y="1" z="1"/>
        <Rotation x="0" y="0" z="0"/>
        <Translation x="0" y="0" z="0"/>
        <Material type="...">...</Material>
    </ObjectName>

    <!-- 光源 (可多个) -->
    <PointLight>...</PointLight>
    <DirectionalLight>...</DirectionalLight>
    <SpotLight>...</SpotLight>
</SceneInfo>
```

### 3.2 全局设置

#### Camera
```xml
<Camera>
    <Args width="256" height="192" outputname="output.png"/>
</Camera>
```
| 属性 | 说明 | 默认值 |
|------|------|--------|
| width | 渲染内部分辨率 | 256 |
| height | 渲染内部分辨率 | 192 |
| outputname | 截图文件名 | output.png |

> 窗口显示分辨率由 `main.cpp` 中的 `Application(1280, 720, ...)` 控制, Camera的width/height决定内部渲染分辨率(影响性能).

#### EnvironmentMap
```xml
<EnvironmentMap>
    <Args display="1"/>
</EnvironmentMap>
```
display=1 时加载 `../../assets/envmap.jpg` 作为天空背景, =0 时无天空(黑色背景).

#### ResourceLimits
```xml
<ResourceLimits>
    <Args maxInstances="16" maxMaterials="16" maxLights="8"/>
</ResourceLimits>
```
限制GPU buffer大小, 根据场景需要调整.

#### 全局光照参数
```xml
<Strength>
    <diffuseStrength value="0.5"/>
    <specularStrength value="1"/>
</Strength>
```

#### 环境光
```xml
<AmbientAgrs>
    <Strength value="0.1"/>
    <Color r="0.1" g="0.1" b="0.1"/>
</AmbientAgrs>
```

#### 其他
```xml
<RefractionMethod> <Args version="1"/> </RefractionMethod>
<DepthMax> <Args DepthMax="4"/> </DepthMax>
<SpheresDisplay> <Args sphere1="0" sphere2="0" sphere3="0" sphere4="0"/> </SpheresDisplay>
```
SpheresDisplay 值为 0/1, 控制4个调试球体是否显示.

---

### 3.3 物体 (Object)

标签名任意(用于标识), 每个物体包含:

```xml
<MyObject>
    <Args filename="model.obj" display="1" normalinterpolation="0"/>
    <Scale x="1.0" y="1.0" z="1.0"/>
    <Rotation x="0.0" y="0.0" z="0.0"/>       <!-- 欧拉角(度)  -->
    <Translation x="0.0" y="0.0" z="0.0"/>
    <Material type="lambertian"> ... </Material>
</MyObject>
```

| Args属性 | 说明 | 默认值 |
|----------|------|--------|
| filename | OBJ模型文件名 (位于 `../../assets/`) | 无 |
| display | 是否显示 (0/1) | 1 |
| normalinterpolation | 是否使用平滑法线 (0=平面, 1=平滑) | 0 |

OBJ文件需放在 `assets/` 目录下, 支持三角形网格.

---

### 3.4 材质 (Material)

#### 通用属性

所有材质类型都需要:
```xml
<Albedo r="0.8" g="0.8" b="0.8"/>
<Roughness value="10.0"/>
```
- `Albedo`: 基础颜色 (线性sRGB, 0-1)
- `Roughness`: 粗糙度指数 (越大越光滑)

#### 漫反射 (lambertian)
```xml
<Material type="lambertian">
    <Albedo r="0.26" g="0.97" b="0.35"/>
    <Roughness value="10.0"/>
</Material>
```
无其他特殊参数.

#### 棋盘格 (checkerboard)
```xml
<Material type="checkerboard">
    <Albedo r="0.8" g="0.8" b="0.8"/>
    <Roughness value="10.0"/>
</Material>
```
程序化棋盘纹理 (XZ平面, 格子0.5m), Albedo的两个颜色自动取白色和深色.

#### 金属 (metal)
```xml
<Material type="metal">
    <Albedo r="0.95" g="0.70" b="0.25"/>     <!-- 金色 -->
    <Roughness value="1425.0"/>
</Material>
```
Albedo的RGB通道分别控制R/G/B波段的Fresnel反射率, 产生彩色金属效果.

#### 介质/玻璃 (dielectric)
```xml
<Material type="dielectric">
    <DispersionA value="1.5"/>           <!-- IOR (折射率) -->
    <DispersionB value="0.004"/>         <!-- Cauchy色散 B -->
    <Reflectivity value="1.0"/>          <!-- Fresnel反射率倍率 -->
    <AbsorptionA value="0.002"/>         <!-- 体积吸收 A (m⁻¹) -->
    <AbsorptionB value="0.0"/>           <!-- 体积吸收 B (m⁻¹·μm²) -->
    <Albedo r="0.6" g="0.7" b="0.8"/>
    <Roughness value="125.0"/>
</Material>
```

| 参数 | 说明 | 典型值 |
|------|------|--------|
| DispersionA | 折射率 n_D | 1.33(水), 1.5(玻璃), 2.4(钻石) |
| DispersionB | Cauchy色散系数 | 0.004(玻璃), 0.014(钻石) |
| Reflectivity | Fresnel反射偏置 | 1.0=物理, >1增强, <1减弱 |
| AbsorptionA | 波长无关吸收系数 α(λ)=A+B/λ² | 0=透明, 0.01=轻微着色 |
| AbsorptionB | 短波吸收增强 (1/λ²项) | 0=均匀, 0.01=暖色玻璃 |

**IOR参考值:**
| 材质 | IOR |
|------|-----|
| 水 | 1.33 |
| BK7光学玻璃 | 1.52 |
| 冕牌玻璃 | 1.5 |
| 钻石 | 2.42 |

---

### 3.5 光源

#### 点光源 (PointLight)
```xml
<PointLight>
    <Position x="2.0" y="-2.0" z="-7.0"/>
    <Color r="1.0" g="1.0" b="1.0"/>
    <Args intensity="5.0" maxDistance="50.0"/>
</PointLight>
```

#### 聚光灯 (SpotLight)
```xml
<SpotLight>
    <Position x="2.0" y="-2.0" z="-7.0"/>
    <Direction x="-1.0" y="-0.5" z="0.0"/>
    <Args innerAngle="2.5" outerAngle="5.5" intensity="2.0" maxDistance="50.0"/>
    <Color r="1.0" g="1.0" b="1.0"/>
</SpotLight>
```
| 参数 | 说明 |
|------|------|
| innerAngle | 内锥角 (度), 内部均匀亮度 |
| outerAngle | 外锥角 (度), 边缘衰减到0 |

#### 平行光 (DirectionalLight)
```xml
<DirectionalLight>
    <Direction x="-0.5" y="-1.0" z="-0.5"/>
    <Color r="1.0" g="0.95" b="0.9"/>
    <Intensity value="5.0"/>
</DirectionalLight>
```

> 光源强度影响光子发射量: 总光子数524288按光源强度比例分配.

---

## 四、完整示例

最小场景 (一个漫反射球 + 一个点光源):

```xml
<?xml version="1.0" encoding="utf-8"?>
<SceneInfo>
    <Camera>
        <Args width="256" height="192" outputname="output.png"/>
    </Camera>
    <EnvironmentMap>
        <Args display="0"/>
    </EnvironmentMap>
    <ResourceLimits>
        <Args maxInstances="4" maxMaterials="4" maxLights="2"/>
    </ResourceLimits>
    <DepthMax> <Args DepthMax="4"/> </DepthMax>
    <SpheresDisplay>
        <Args sphere1="1" sphere2="0" sphere3="0" sphere4="0"/>
    </SpheresDisplay>

    <!-- 调试球体使用内建shader, 不需要OBJ文件 -->
    <!-- 自定义物体需要OBJ: -->
    <Bunny>
        <Args filename="bunny-mesh.obj" display="1" normalinterpolation="1"/>
        <Scale x="0.3" y="0.3" z="0.3"/>
        <Translation x="0" y="-1.0" z="-2.0"/>
        <Material type="lambertian">
            <Albedo r="0.8" g="0.8" b="0.8"/>
            <Roughness value="10.0"/>
        </Material>
    </Bunny>

    <PointLight>
        <Position x="2.0" y="-2.0" z="-7.0"/>
        <Color r="1.0" g="1.0" b="1.0"/>
        <Args intensity="5.0" maxDistance="50.0"/>
    </PointLight>

    <Strength>
        <diffuseStrength value="0.5"/>
        <specularStrength value="1"/>
    </Strength>
    <AmbientAgrs>
        <Strength value="0.1"/>
        <Color r="0.1" g="0.1" b="0.1"/>
    </AmbientAgrs>
</SceneInfo>
```

---

## 五、注意事项

1. **首次运行**: 窗口可能很暗 — 光子图需要多帧累积, 等30-60帧后亮度正常.
2. **截图路径**: 截图PNG保存在exe同级目录, 文件名由 `<Camera outputname="..."/>` 决定.
3. **OBJ格式**: 仅支持三角形网格 (不支持四边形), 需要法线数据.
4. **修改配置**: 编辑XML后需重启程序, 不支持热重载.
5. **性能**: 金属和玻璃材质比漫反射贵很多 (多次弹射+色散), 场景中适度使用.
