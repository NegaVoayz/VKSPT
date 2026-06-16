# 光谱颜色转换设计

RGB↔连续光谱的双向转换: Wiener基重建、Legendre展开反射率、桶CMF投影显示。

---

## 一、数据基础 (`cmf_data.slang`)

### 1.1 CMF采样

CIE 1931 2° 标准观察者 → XYZ→sRGB线性矩阵 → 81采样点 (380–780nm, Δ5nm):

- `CMF_R[81]`, `CMF_G[81]`, `CMF_B[81]` — 直接到线性sRGB的CMF值(非XYZ)
- 每条CMF独立归一化

### 1.2 Wiener基函数 (RGB→光谱)

由sRGB→XYZ→CMF伪逆计算, 用于从RGB线性值重建连续光谱:

```
S(λ) = lin_R · BASIS_R(λ) + lin_G · BASIS_G(λ) + lin_B · BASIS_B(λ)
```

- `BASIS_R[81]`, `BASIS_G[81]`, `BASIS_B[81]` — 每个基函数81采样点
- `estimateSpectrum(lamNm, linearRGB)` — 对任意波长λ, 线性插值基函数后点积

### 1.3 桶预积分CMF (光谱→sRGB)

`BUCKET_CMF[10]` — 每个40nm光谱桶的预积分CMF, 直接到线性sRGB:

```
BUCKET_CMF[b] = ∫_{λ_b}^{λ_b+40} CMF(λ) dλ
```

其中 `CMF(λ)` 为线性sRGB空间的color matching function. 用于显示时的快速光谱→RGB转换.

---

## 二、Gauss-Legendre求积

### 2.1 3点求积 (Legendre投影 / 吸收)

用于在波长区间上做2阶Legendre展开的数值积分:

| k | 节点 x_k | 权重 w_k |
|---|---------|---------|
| 0 | -√(3/5) ≈ -0.77459667 | 5/9 |
| 1 | 0 | 8/9 |
| 2 | +√(3/5) ≈ +0.77459667 | 5/9 |

波长映射:

```
λ_k = λ_mid + halfWidth · x_k
```

积分近似:

```
∫_{-1}^{1} f(x) dx ≈ Σ_k w_k · f(x_k)
```

由于GL求积在[-1,1]上归一化, 实际积分需乘 `halfWidth` (或结果乘0.5后再乘区间宽度).

### 2.2 7点求积 (宽带颜色积分)

用于天空/光源等宽带(>120nm)的颜色积分(`integrateSkySRGB`):

| k | x_k | w_k |
|---|-----|-----|
| 0 | -0.94910791 | 0.12948497 |
| 1 | -0.74153119 | 0.27970539 |
| 2 | -0.40584515 | 0.38183005 |
| 3 | 0.0 | 0.41795918 |
| 4 | +0.40584515 | 0.38183005 |
| 5 | +0.74153119 | 0.27970539 |
| 6 | +0.94910791 | 0.12948497 |

每点评估: `S_k × D_k × CMF_k × halfWidth`, 求和得线性sRGB.

---

## 三、RGB → 光谱 (albedo反射率)

**函数**: `albedoLegendreExpand(alb, lamMid, halfW, out C0, C1, C2)` (`color.slang`)

**用途**: Lambertian/Checkerboard材质的RGB albedo → log-Legendre衰减系数

### 3.1 反射率模型 (CMF加权)

每个波长的表面反射率:

```
R(λ) = dot(alb_rgb, CMF(λ)) / (CMF_R(λ) + CMF_G(λ) + CMF_B(λ))
```

物理直觉: 白色表面 (alb=1,1,1) 在所有λ上R(λ)=1; 纯红表面 (alb=1,0,0) 只在CMF_R主导的λ上反射率高.

CMF分母趋零时的fallback: `R(λ) = dot(alb, 1/3)`.

### 3.2 亮度归一化

在3个GL点上eval线性反射率值 `linVal[k]`:

```
linAvg = ½ Σ_k w_k · linVal[k]    (GL积分均值)
```

归一化因子: `nrm = albLum / linAvg`, 其中 `albLum = dot(alb, 1/3)`.

使GL平均的线性反射率 = albedo亮度. 安全clamp: nrm ∈ [0.1, 10.0].

### 3.3 Legendre投影

```
L[k] = ln(nrm · linVal[k])
```

对3个L值用GL求积公式投影到2阶Legendre多项式:

```
c₀ = ½ (w₀·L₀ + w₁·L₁ + w₂·L₂)
c₁ = 1.5 (w₀·L₀·(-SQ3_5) + w₂·L₂·SQ3_5)
c₂ = 2.5 (w₀·L₀·P₂(-SQ3_5) + w₁·L₁·P₂(0) + w₂·L₂·P₂(SQ3_5))
    = 2.5 (w₀·L₀·0.4 + w₁·L₁·(-0.5) + w₂·L₂·0.4)
```

其中 `P₂(x) = 1.5x² - 0.5`.

---

## 四、RGB → 光谱 (金属Fresnel)

**函数**: `reflectanceLegendreExpand(r0, r1, r2, lamC, lamH, lamMid, halfW, out C0, C1, C2)` (`color.slang`)

**用途**: 金属材质的RGB Fresnel系数 → log-Legendre衰减系数

### 4.1 金属Fresnel的Legendre表示

金属albedo RGB预先在全局光谱范围 [LAMBDA_MIN, LAMBDA_MAX] 上展开为Legendre系数 `(r₀, r₁, r₂)`:

```
R_metal(λ) = evalLegendre(r₀, r₁, r₂, (λ-λ_center) / λ_halfwidth)
```

其中 `λ_center = 580nm`, `λ_halfwidth = 200nm`.

### 4.2 重投影到射线波长区间

在射线的3个GL波长点 `λ_k = lamMid + halfW·x_k` 上:

```
R_k = evalLegendre(r₀, r₁, r₂, (λ_k - λ_center) / λ_halfwidth)
L_k = ln(R_k)
```

然后同样的GL→Legendre投影得到 `(c₀, c₁, c₂)`.

---

## 五、RGB → 光谱 (光源/Wiener基)

**函数**: `estimateSpectrum(lamNm, linearRGB)` (`color.slang`)

**用途**: 光源颜色、天空贴图、光子初始光谱

### 5.1 Wiener基重建

```
S(λ) = lin_R · BASIS_R(λ) + lin_G · BASIS_G(λ) + lin_B · BASIS_B(λ)
```

对81采样点的基函数做线性插值后点积.

### 5.2 光子存储 (`storePhoton` → `photon_hit.slang`)

光子击中漫反射表面时, 每个40nm桶:

```
specBucket[b] = estimateSpectrum(λ_b_center, lightRGB)
              × evalDecay(c₀, c₁, c₂, x_b)   // Legendre衰减
              × (oE - oS)                      // 重叠宽度
```

其中 `x_b = (λ_b_center - λ_mid) / halfW`, clamped to [-1,1].

### 5.3 天空积分 (`accumulateSpectrumSky` → `sky.slang`)

快速路径(无分散, spread2<1e-6): 单envMap采样 + 3点GL求积:
```
spectral[b] += estimateSpectrum(λ_mid, envLinear) × D(0) × (oE-oS)
```

或宽带路径: 对每个bucket的带宽检入统一decay模型:
```
spectral[b] += S × exp(c₀-0.5c₂) × (oE-oS)
```

分散路径(spread2≥1e-6): 每个bucket独立采样envMap:
```
sampleDir[b] = normalize(dir + t(b) × Vperp)
spectral[b] += estimateSpectrum(λ_mid, envMap(sampleDir[b])) × decay(b) × width(b)
```

---

## 六、光谱 → RGB (显示)

**位置**: `RayGenMain` → `raytrace_pipeline.slang`

### 6.1 桶CMF累加

每个光谱桶的贡献直接加到像素线性sRGB:

```
pixelColor = Σ_b (p.spectral[b] / 40.0) × BUCKET_CMF[b]
```

- `p.spectral[b]` — 桶b上累积的光谱能量 (nm·radiance)
- `/40.0` — 除以桶宽度得光谱密度
- `BUCKET_CMF[b]` — 桶b内CMF预积分, 直接到线性sRGB

### 6.2 跨帧累积

```
newAcc = oldAcc + pixelColor
newCnt = oldCnt + spp
display = linearToSRGB(newAcc / newCnt × 2.5)
```

---

## 七、吸收算子 (`applyAbsorption`)

**函数**: `applyAbsorption(lamStart, lamEnd, C0, C1, C2, reflC0, reflC1, reflC2)` (`color.slang`)

**用途**: 在ClosestHitMain中统一应用吸收, 光源Legendre状态 + 反射Legendre系数.

### 7.1 Log空间加法

对3个GL点, 光源衰减和反射率在log空间中相加:

```
newL[k] = (c₀ + c₁·x_k + c₂·(1.5x_k²-0.5))    // 光源衰减
        + (reflC₀ + reflC₁·x_k + reflC₂·(1.5x_k²-0.5))  // 反射率
```

然后重新投影回Legendre系数.

### 7.2 波长裁剪

应用吸收后, 裁剪波长范围两端衰减过大的部分(衰减 < e^(-13.8155) ≈ 1e-6):

```
while lamStart < lamEnd:
  if evalDecay(c₀,c₁,c₂, x_start) >= 1e-6: break
  lamStart += max(1, (lamEnd-lamStart)/8)
```

对lamEnd同理反向裁剪.

---

## 八、玻璃体积吸收

**位置**: `handleDielectric` → `material_handlers.slang`

**模型**: Beer-Lambert透过率 `T(λ,d) = exp(-α(λ)·d)`, log空间加法.

### 8.1 数据结构

`GpuMaterial`中 `absorpA.xyz` / `absorpB.xyz` — Cauchy吸收系数(标量,存于float4各通道取均值):
- A: 波长无关吸收 (m⁻¹)
- B: 1/λ²系数 (m⁻¹·μm²), 短波吸收更强

**α(λ) = A + B/λ²** 是完整的光谱吸收曲线, 不依赖RGB通道.

### 8.2 Legendre拟合

复用 `applyVolumeDecay()` (`legendre_decay.slang`):
```
在3个GL点 λ_k:  aVal[k] = -(A + B/λ_k²) × hitT
→ GL求积投影 → (a₀, a₁, a₂)
throughput += a₀; decayC1 += a₁; decayC2 += a₂
```

**触发条件**: `entering=false` (射线在玻璃内击中出射面), `absA>0||absB>0`.
