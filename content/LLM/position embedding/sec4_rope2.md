---
title: 第 4 节 旋转位置编码-2
---

| [[sec4_rope1\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope3\|下一节]] |
| :-----------------: | :----------------------------------: | :----------------------: |

在 [[sec4_rope1\|上一节]] 中，我们介绍了标准的 [[sec4_rope1\|RoPE]]，它对所有维度施加旋转。然而，全量旋转是否总是最优？本节我们将探讨 **Partial RoPE（$\partial$-RoPE）**——一种仅在部分维度上应用旋转的编码策略，它不仅在效果上优于全量 RoPE，更是 MLA（Multi-head Latent Attention）等高效架构的关键技术。

## 动机：语义与位置的平衡

### 完全 RoPE 的局限

标准 [[sec4_rope1\|RoPE]] 的表达式（参见 [[sec4_rope1#RoPE 矩阵形式\|公式 (1)]]）为：

$$
\boldsymbol{q}_i = \boldsymbol{x}_i\boldsymbol{W}_q\boldsymbol{\mathcal{R}}_i, \quad \boldsymbol{k}_j = \boldsymbol{x}_j\boldsymbol{W}_k\boldsymbol{\mathcal{R}}_j.
$$

这种「全维度参与」的设计虽然保证了位置信息的完整注入，但也可能带来一个问题：**过度强调位置可能干扰语义信息的表达**。

从 [[kexue.fm/10122\|RoPE 底数选择原则]] 的「语义聚合」视角来看，我们希望当 $\boldsymbol{k}$ 与 $\boldsymbol{q}$ 语义相近时，无论距离多远，注意力都应该较大。这要求：

$$
\sum_{m=0}^{d/2-1} \cos (i-j)\theta_m \geq 0.
$$

有趣的是，如果我们**让部分维度不旋转**（即 $\theta=0$，$\cos(0)=1$），上述不等式将更容易满足。

## Partial-RoPE 的定义

### 形式化表达

$\partial$-RoPE 将维度分为两部分：

> - **旋转部分**（$d_r$ 维）：应用标准 RoPE；
> - **固定部分**（$d_c$ 维）：不施加位置编码（NoPE）。

^eq1
$$
\boldsymbol{q}_i = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc},\; \boldsymbol{x}_i\boldsymbol{W}_{qr}\boldsymbol{\mathcal{R}}_i\right], \tag{1}
$$

^eq2
$$
\boldsymbol{k}_j = \left[\boldsymbol{x}_j\boldsymbol{W}_{kc},\; \boldsymbol{x}_j\boldsymbol{W}_{kr}\boldsymbol{\mathcal{R}}_j\right]. \tag{2}
$$

内积分解为：

$$
\boldsymbol{q}_i^\top\boldsymbol{k}_j = \underbrace{\boldsymbol{x}_i\boldsymbol{W}_{qc}\boldsymbol{W}_{kc}^\top\boldsymbol{x}_j^\top}_{\text{语义项}} + \underbrace{(\boldsymbol{x}_i\boldsymbol{W}_{qr})^\top\boldsymbol{\mathcal{R}}_{j-i}(\boldsymbol{x}_j\boldsymbol{W}_{kr})}_{\text{位置项}}.
$$

这实现了**显式的语义-位置解耦**。

## 与 MLA 的结合

$\partial$-RoPE 的价值在 MLA 中得到了充分体现。MLA 通过低秩投影压缩 KV Cache，但标准 RoPE 会阻碍矩阵合并：

$$
\boldsymbol{q}_i^\top\boldsymbol{k}_j = \boldsymbol{x}_i\boldsymbol{W}_q\boldsymbol{\mathcal{R}}_{i-j}\boldsymbol{W}_k^\top\boldsymbol{c}_j^\top,
$$

其中 $\boldsymbol{\mathcal{R}}_{i-j}$ 与位置相关，无法合并为固定矩阵。

采用 $\partial$-RoPE 后，无旋转部分可以吸收到投影矩阵中（对比 [[sec4_rope1#高效实现\|标准 RoPE]] 的形式）：

^eq3
$$
\boldsymbol{q}_i = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc},\; \boldsymbol{x}_i\boldsymbol{W}_{qr}\boldsymbol{\mathcal{R}}_i\right], \quad \boldsymbol{k}_j = \left[\boldsymbol{c}_j\boldsymbol{W}_{kc},\; \boldsymbol{x}_j\boldsymbol{W}_{kr}\boldsymbol{\mathcal{R}}_j\right]. \tag{3}
$$

这使得 MLA 在推理时可以保持 KV Cache 的压缩优势。

## 实验验证

在约 1B 参数的模型上对比（KV Cache 固定为 512）：

| 配置 | Loss | 备注 |
| :--- | :--: | :--- |
| GQA2-128 | 2.750 | $d_k=128$，标准 RoPE |
| GQA1-256 | 2.720 | $d_k=256$，标准 RoPE |
| **GQA1-256-PR** | **2.711** | **$d_k=256$，$\partial$-RoPE** |
| MLA | 2.721 | $d_c=128, d_r=64$ |

结果排序：

$$
\text{GQA2-128} < \text{MLA} \lesssim \text{GQA1-256} < \text{GQA1-256-PR}.
$$

这表明：**增大 head_dims 收益显著，而 $\partial$-RoPE 能带来额外增益**。

## 理论意义

### 底数约束的放宽

RoPE 底数 $b$ 需满足：

$$
\sum_{i=0}^{d/2-1} \cos m\theta_i \geq 0, \quad m \in [0, L-1].
$$

采用 $\partial$-RoPE 后变为：

$$
d_c + \sum_{i=0}^{d_r/2-1} \cos m\theta_i \geq 0.
$$

由于 $d_c > 0$，这一不等式**自动满足**，从根本上放宽了底数与训练长度的耦合。

### 维度分配建议

| 方法 | $d_c$（固定） | $d_r$（旋转） | 比例 |
| :--- | :-----------: | :-----------: | :--: |
| MLA-V2 | 128 | 64 | 2:1 |
| MLA-256 | 192 | 64 | 3:1 |

实践上，旋转比例 $1/4 \sim 1/3$ 是较好的选择。

## 小结

$\partial$-RoPE 作为 [[sec4_rope1\|RoPE]] 的改进版本，通过在部分维度上保持固定，实现了语义与位置的显式解耦。它不仅在实验中优于全量 [[sec4_rope1\|RoPE]]，更是 [[kexue.fm/10091\|MLA]] 等高效架构的关键技术。在 [[sec4_rope3\|下一节]] 中，我们将探讨另一种 RoPE 变体——[[sec4_rope3\|VO-RoPE]]，它通过 Value 和 Output 的旋转提供了不同的实现视角。

| [[sec4_rope1\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope3\|下一节]] |
| :-----------------: | :----------------------------------: | :----------------------: |
