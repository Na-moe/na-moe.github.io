---
title: 第 4 节 旋转位置编码-1
---

| [[sec3_rel_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope2\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |

在前面的章节中，我们介绍了从绝对位置编码到相对位置编码的发展。然而，这些编码方式大多需要在 Attention 中引入额外的位置偏置项。本节我们将介绍 **RoPE（Rotary Position Embedding）**，一种通过旋转操作将绝对位置编码自然转化为相对位置编码的优雅方案。

## 第一性原理推导

RoPE 的核心目标可以概括为两点：
1. 给 $\boldsymbol{q}, \boldsymbol{k}$ 注入**绝对位置**信息；
2. 使 $\boldsymbol{q}^\top\boldsymbol{k}$ 的结果带有**相对位置**信息。

### 形式化定义

假设通过函数 $\boldsymbol{f}: \mathbb{R}^{d} \times [\![1, L]\!] \mapsto \mathbb{R}^{d}$ 编码绝对位置：

$$
\tilde{\boldsymbol{q}}_{i} = \boldsymbol{f}\!\left( \boldsymbol{q}, i \right), \quad \tilde{\boldsymbol{k}}_{j} = \boldsymbol{f}\!\left( \boldsymbol{k}, j \right).
$$

我们希望内积满足：

$$
\langle \tilde{\boldsymbol{q}}_{i}, \tilde{\boldsymbol{k}}_{j} \rangle = g\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right),
$$

其中 $g$ 是仅依赖于相对距离 $i-j$ 的函数。

### 二维情形的求解

类似于 [[sec2_abs_pe#Sin.PE 的简化分析|Sin.PE]] 的做法，先考虑二维情形。将向量视为复数，我们希望：

$$
\mathfrak{Re}\!\left[ \boldsymbol{f}\!\left( \boldsymbol{q}, i \right) \boldsymbol{f}^*\!\left( \boldsymbol{k}, j \right) \right] = g\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right).
$$

设指数形式：

$$
\boldsymbol{f}\!\left( \boldsymbol{q}, i \right) = R_{f}\!\left( \boldsymbol{q}, i \right) \mathrm{e}^{ \mathrm{i} \Theta_{f}\left( \boldsymbol{q}, i \right) },
$$

由 $\boldsymbol{f}(\boldsymbol{q}, 0) = \boldsymbol{q}$ 可得边界条件。代入 $j=i$ 和 $j=i-1$ 分别推导：

- **模长守恒**：$R_{f}(\boldsymbol{q}, i) = \lVert \boldsymbol{q} \rVert$，位置编码不应改变向量模长；
- **角度叠加**：$\Theta_{f}(\boldsymbol{q}, i) = \Theta(\boldsymbol{q}) + i\theta$，即每增加一个位置，角度增加固定值 $\theta$。

### RoPE 矩阵形式

综上，二维情形的解为：

$$
\boldsymbol{f}\!\left( \boldsymbol{q}, i \right) = \begin{pmatrix}
\cos i\theta & -\sin i\theta \\
\sin i\theta & \cos i\theta
\end{pmatrix} \begin{pmatrix} q_{1} \\ q_{2} \end{pmatrix}.
$$

扩展到 $d$ 维（偶数），RoPE 是一个分块对角矩阵：

^eq1
$$
\boldsymbol{f}\!\left( \boldsymbol{q}, i \right) = \boldsymbol{\mathcal{R}}_{i} \boldsymbol{q} = \begin{pmatrix}
\cos i\theta_{1} & -\sin i\theta_{1} & \cdots & 0 & 0 \\
\sin i\theta_{1} & \cos i\theta_{1} & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & \cos i\theta_{d/2} & -\sin i\theta_{d/2} \\
0 & 0 & \cdots & \sin i\theta_{d/2} & \cos i\theta_{d/2}
\end{pmatrix} \begin{pmatrix} q_{1} \\ q_{2} \\ \vdots \\ q_{d-1} \\ q_{d} \end{pmatrix}, \tag{1}
$$

其中频率 $\theta_{i} = 10000^{-2i/d}$，从 $1$ 渐变到接近 $0$。

### 相对位置的内积

利用旋转矩阵的性质 $\boldsymbol{\mathcal{R}}_{i}^\top \boldsymbol{\mathcal{R}}_{j} = \boldsymbol{\mathcal{R}}_{j-i}$，可得：

$$
(\boldsymbol{\mathcal{R}}_{i}\boldsymbol{q})^\top (\boldsymbol{\mathcal{R}}_{j}\boldsymbol{k}) = \boldsymbol{q}^\top \boldsymbol{\mathcal{R}}_{j-i} \boldsymbol{k}.
$$

这证明了 RoPE 天然具有**相对位置编码**的特性！

## 高效实现

RoPE 可以通过元素级操作高效实现，无需构造完整矩阵：

```python
def rope(x: Tensor['seq', 'd'], freq: float = 10000.0) -> Tensor['seq', 'd']:
    seq_len, d = x.shape
    
    # 位置与频率
    pos = torch.arange(seq_len, dtype=float, device=x.device)[:, None]
    theta = torch.exp(math.log(freq) * -torch.arange(0, d, 2) / d)[None, :]
    
    cos, sin = torch.cos(pos * theta), torch.sin(pos * theta)
    
    # 交替应用
    x1, x2 = x[:, 0::2], x[:, 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

## 远程衰减性质

RoPE 的内积具有**远程衰减**特性。考虑 $(\boldsymbol{\mathcal{R}}_{i}\boldsymbol{q})^\top(\boldsymbol{\mathcal{R}}_{j}\boldsymbol{k})$：

$$
= \mathfrak{Re}\!\left[ \sum_{k=1}^{d/2} \boldsymbol{q}_{[2k-1:2k]} \boldsymbol{k}^*_{[2k-1:2k]} \mathrm{e}^{\mathrm{i}(i-j)\theta_{k}} \right].
$$

通过 Abel 分部求和法可以证明，当 $|i-j|$ 增大时，内积的绝对值趋于衰减。这意味着位置相近的 Token 会获得更大的注意力权重，符合我们的直观期望。

![[rope_attenuation.png]]

## 小结

RoPE 的核心优势在于：

1. **数学优雅**：通过旋转操作，将绝对位置编码自然转化为相对位置编码；
2. **实现高效**：仅需元素级操作，无需额外参数；
3. **性质优良**：保持模长不变，具有远程衰减特性。

RoPE 已成为现代 LLM 的主流位置编码方案。在后续章节中，我们将探讨如何在此基础上进一步优化——包括部分维度的旋转（[[sec4_rope2\|Partial RoPE]]）、Value 侧的旋转（[[sec4_rope3\|VO-RoPE]]），以及长度外推方法（[[sec4_rope4\|YaRN]]）。

| [[sec3_rel_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope2\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |
