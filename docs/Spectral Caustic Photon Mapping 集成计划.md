# Spectral Caustic Photon Mapping 集成计划

## 目标

在保持现有 Spectral Packet / Legendre / D-V Dispersion 系统不变的前提下，引入 Caustic Photon Mapping，以解决以下问题：

- 钻石火彩
- 棱镜投影彩虹
- 玻璃焦散
- 高阶折射链导致的光源命中概率过低问题

不引入 BDPT、VCM、MLT 等复杂路径积分算法。

第一阶段目标仅为获得稳定、可见的彩色焦散。

------

# 核心思想

现有系统：

Camera
→ Scene
→ Light

扩展为：

Light
→ Scene
→ Photon Map

以及：

Camera
→ Scene
→ Photon Gather

Photon Tracing 与现有 Ray Tracing Pipeline 共用：

- BVH
- Intersection
- Material Evaluation
- Dispersion Propagation
- Packet Split
- Legendre Restriction

------

# 光源表示

保持现有资源接口：

- RGB Emissive
- Directional Light
- HDRI
- Temperature Light

内部统一转换为：

Legendre Spectrum

即：

RGB
→ Spectrum Reconstruction
→ Legendre Projection

结果作为 Photon 初始能量。

------

# Photon 能量表示

Photon 内部维护：

Φ(λ)

即当前剩余能量。

推荐传输阶段使用：

g(λ)=ln(Φ(λ))

进行累计。

即：

g += absorption

g += medium attenuation

g += any logarithmic decay term

避免频繁乘法。

Photon 存储前：

Φ = exp(g)

重新投影回 Legendre 系数。

------

# Split 规则

完全复用当前系统。

传播过程中维护：

- Dispersion Distance D
- Dispersion Velocity V

当：

|D|

超过 force_split_width

触发回溯 Split。

使用现有：

2^k Restriction

流程。

Photon Mapping 不增加新的 Split 逻辑。

------

# Photon 存储条件

第一阶段仅实现 Caustic Photon Map。

Photon 满足：

specular_chain_length > 0

且当前命中：

Diffuse Surface

时存储。

例如：

Sun
→ Diamond
→ Diamond
→ Wall

存储。

而：

Sun
→ Wall

不存储。

目的：

仅保留焦散贡献。

减少 Photon 数量。

------

# Photon Record

Photon Map 中记录：

Position

Surface Normal

Incoming Direction

Lambda Range

Legendre Energy Coefficients

其中：

Legendre Energy Coefficients

表示当前 Photon 携带的实际光谱能量。

不记录：

- 光源 ID
- D
- V

第一阶段不需要。

后续如需高级重建再扩展。

------

# 第一阶段 Gather

不实现 KD Tree。

不实现 Hash Grid。

不实现 Radius Optimization。

直接：

PhotonBuffer

顺序遍历。

对于命中点：

for photon in PhotonBuffer

计算：

distance

若：

distance < gather_radius

则累加贡献。

目标：

验证 Spectral Photon 流程正确。

验证彩虹和焦散能够形成。

性能暂不作为目标。

------

# 第二阶段 Gather

在验证正确后：

引入 Spatial Hash。

替代全表扫描。

支持：

- 邻域查询
- 固定半径 Gather

不改变 Photon 数据结构。

------

# 第三阶段

引入：

KD Tree

或

Uniform Grid

优化。

仅优化查询性能。

不改变光谱表示。

------

# 验证场景

优先级：

1. 三棱镜 + 太阳

验证：

- 色散方向
- Split
- Legendre Restriction

------

1. 玻璃球 + 太阳

验证：

- 焦散形成
- 光谱连续性

------

1. 钻石 + 太阳

验证：

- 多次内部反射
- Fire
- 高频 Split

------

1. HDRI + 钻石

验证：

- 环境照明下焦散稳定性

------

# 暂不实现

以下内容明确延后：

- BDPT
- VCM
- MLT
- Specular Manifold Sampling
- Progressive Photon Mapping
- Vertex Merging
- Spectral MIS

避免项目复杂度失控。

------

# 成功标准

以下任一结果出现即视为阶段成功：

- 棱镜在墙面投射连续彩虹
- 玻璃球形成彩色焦散
- 钻石产生可见 Fire
- 焦散区域光谱连续，无明显分段边界

此时再考虑性能优化与高级积分方法。