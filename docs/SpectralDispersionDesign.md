# 光谱色散分裂算法设计

## 状态

每个光谱包维护：

$$
[\lambda_B,\lambda_R]
$$

中心射线：

$$
(x,\omega)
$$

色散状态（偏移向量，偏移速度）：

$$
D,V
$$

Legendre 系数：

$$
(c_0,c_1,c_2)
$$

---

# 传播

传播到下一次 hit：

$$
D \leftarrow D+tV
$$

其中 (t) 为本段 march distance。

---

计算红蓝跨度：

$$
d = 2|D|
$$

（如果只关心横向尺寸，就先投影到 (\omega) 的垂面）

---

# 判定分裂

给定阈值：

$$
W
$$

若：

$$
d \le W
$$

继续追踪。

---

否则：

$$
k
=

\left\lceil
\log_2
\frac dW
\right\rceil
$$

生成：

$$
n=2^k
$$

个子包。

当前包终止。

---

# 预计算

定义：

$$
s = 2^{-k}
$$

$$
s_2 = 4^{-k}
$$

---

Legendre 缩放：

$$
c_2^{scale}
=

s_2 c_2
$$

$$
c_1^{base}
=

s c_1
$$

---

子包色散：

$$
D_{child}=sD
$$

$$
V_{child}=sV
$$

---

# 子包中心参数

第 (i) 个子包：

$$
i=0,\dots,n-1
$$

中心坐标：

$$
S_i
=

-(1-s)
+
2is
$$

等价递推：

$$
S_0=-(1-s)
$$

$$
S_{i+1}=S_i+2s
$$

---

# 子包几何状态

位置：

$$
x_i
=

x + S_iD
$$

方向：

$$
\omega_i
=

\operatorname{normalize}
(
\omega + S_iV
)
$$

---

# 子包波长区间

总宽度：

$$
L=\lambda_R-\lambda_B
$$

使用 Bresenham 式递推：

```cpp
acc += L;
right = lambdaB + (acc >> k);
```

自动均匀分配余数。

---

# 子包 Legendre 系数

定义：

$$
Q_i=S_i^2
$$

---

$$
c_{2,i}
=

s_2 c_2
$$

---

$$
c_{1,i}
=

s(c_1+3S_i c_2)
$$

---

$$
c_{0,i}
=

c_0
+
S_i c_1
+
\left(
\frac32Q_i
-\frac12
+\frac12 s_2
\right)c_2
$$

---

# 高效递推

避免每个子包重新算平方。

设：

$$
\Delta = 2s
$$

---

初始化：

$$
S=-(1-s)
$$

$$
Q=S^2
$$

---

每次循环：

$$
Q \leftarrow Q + 2S\Delta + \Delta^2
$$

$$
S \leftarrow S+\Delta
$$

---

# 最终伪代码

```cpp
trace(packet)
{
    hit();

    D += t * V;

    d = 2 * length(D);

    if(d <= SplitWidth)
        continue;

    k = ceil_log2(d / SplitWidth);

    n = 1 << k;

    s  = exp2(-k);
    s2 = exp2(-2*k);

    childD = D * s;
    childV = V * s;

    S     = -(1 - s);
    Delta = 2 * s;
    Q     = S * S;

    initLambdaGenerator();

    for(i=0;i<n;i++)
    {
        child.pos =
            pos + S * D;

        child.dir =
            normalize(dir + S * V);

        child.D = childD;
        child.V = childV;

        child.c2 =
            s2 * c2;

        child.c1 =
            s * (c1 + 3*S*c2);

        child.c0 =
            c0
            + S*c1
            + (1.5*Q - 0.5 + 0.5*s2)*c2;

        child.lambdaRange =
            nextLambdaInterval();

        enqueue(child);

        Q += 2*S*Delta + Delta*Delta;
        S += Delta;
    }

    kill(parent);
}
```

这套算法的核心特点是：

* 分裂层数只由一次 `log2` 决定；
* 所有区间边界用整数递推生成；
* Legendre 重投影不需要矩阵乘法；
* 每个子包的系数直接由 (S_i) 闭式计算；
* 生成 (2^k) 个子包时不需要递归 restriction；
* 除法全部消失，只剩若干 (2^{-k}) 缩放和 FMA。
