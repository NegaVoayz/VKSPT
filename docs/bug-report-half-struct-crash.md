# Bug: half类型在compute shader新binding槽导致驱动崩溃

## 症状
进程退出码 `0xC0000409` (`STATUS_STACK_BUFFER_OVERRUN`)，发生在渲染循环中。

## 复现
1. 在 `common.bindings.slang` 新增binding:
```hlsl
public [[vk::binding(18, 0)]] RWStructuredBuffer<PhotonRecord> cellPhotonBuffer;
```
2. Compute shader写入该buffer → 崩溃。

## 隔离测试

| 测试 | 结果 |
|------|------|
| Compute写 `RWStructuredBuffer<PhotonRecord>` 到新binding 18 | 崩溃 |
| Compute写 `uint` 到 `RWStructuredBuffer<uint>` (binding 17) | 正常 |
| Compute读 `RWStructuredBuffer<PhotonRecord>` 到已有binding 7 | 正常 |
| 同一dispatch用已知好的SPV (scatter) | 正常 |

## 根因假设
仅当**新的**compute pipeline访问**新的**binding槽且`RWStructuredBuffer<T>`的`T`包含`half`类型时触发。已有binding 7 (`photonBuffer`)同类型在RT和compute管线均正常。

可能的驱动层问题: SPIR-V模块只引用一个含half类型的StorageBuffer变量时，驱动处理稀疏binding布局异常。

## 解决方案
使用 `RWStructuredBuffer<float>` 手动打包:

```hlsl
public [[vk::binding(18, 0)]] RWStructuredBuffer<float> cellPhotonData;

// 每cell 23 floats:
// [0..3]   position (xyz + pad)
// [4..7]   normal (xyz + pad)
// [8..11]  direction (xyz + pad)
// [12..21] spectral buckets 0..9
// [22]     photon count

// 写入:
uint base = hash * 23;
cellPhotonData[base + 0] = pos.x; ...

// 读取:
float3 pos = float3(cellPhotonData[base+0], cellPhotonData[base+1], cellPhotonData[base+2]);
```

## 环境
- Vulkan SDK 1.4.341.1, Slang (spirv), GPU: RTX 4060
- 启用的features: `storageBuffer16BitAccess`, `shaderFloat16`, `bufferDeviceAddress`
