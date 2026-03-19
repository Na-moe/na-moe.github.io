---
title: 第 4.4 节 旋转位置编码-4
---

| [[sec4_rope3\|上一节]] | [[LLM/position embedding/index\|目录]] | —— |
| :-----------------: | :----------------------------------: | :-: |

在前面三节中，我们深入探讨了 [[sec4_rope1\|RoPE]] 及其变体（[[sec4_rope2\|Partial RoPE]]、[[sec4_rope3\|VO-RoPE]]）的理论基础。本节我们将关注一个实际应用问题：**长度外推**——如何让在短文本上训练的模型能够处理更长的上下文？我们将介绍 [**YaRN** (Yet another RoPE extensioN)](https://arxiv.org/abs/2309.00071)，一种基于「转圈视角」的高效长度外推方法。

## 动机：转圈视角

### OOD 问题的本质

RoPE 通过复数旋转编码位置：对于维度 $i$，旋转角度为 $\theta_i = 10000^{-2i/d}$（参见 [[sec4_rope1#RoPE 矩阵形式\|第一节]]）。当相对距离 $m-n$ 变化时，单位圆上的点随之转动：

- **高频**（$\theta_i$ 较大）：转速快，周期短
- **低频**（$\theta_i$ 较小）：转速慢，周期长

假设训练长度为 $L_{\text{train}}$，关键观察是：

> **不是位置编号 $m-n$ 是否 OOD，而是单位圆上的点是否被充分训练过。**

对于高频维度，在训练范围内已转多圈，圆上各点都被见过；对于低频维度，可能只转了部分弧，测试时若超出此范围则表现不可预测。

## YaRN 核心方法

### 圈数计算

维度 $i$ 的周期和圈数：

$$
T_i = \frac{2\pi}{\theta_i}, \quad r_i = \frac{L_{\text{train}}}{T_i} = \frac{\theta_i L_{\text{train}}}{2\pi}.
$$

### 动态内插策略

基于圈数 $r_i$ 制定策略：

1. **$r_i > \tau$**：充分训练，**不做改动**（直接外推）
2. **$r_i < 1$**：需**内插**，将 $\theta_i$ 缩放 $\frac{L_{\text{train}}}{L_{\text{test}}}$
3. **其他情况**：线性过渡

^eq1
$$
\theta_i^{\text{new}} = \left[\gamma_i + (1-\gamma_i)\frac{L_{\text{train}}}{L_{\text{test}}}\right]\theta_i, \tag{1}
$$

其中

$$
\gamma_i = \begin{cases} 
1, & r_i > \tau \\ 
0, & r_i < 1 \\ 
\frac{r_i - 1}{\tau - 1}, & \text{others}
\end{cases}
$$

这就是 YaRN 的核心——**高频外推、低频内插**。

### Attention Scale 因子

YaRN 还引入一个经验性 Scale 因子：

^eq2
$$
\lambda = \left(1 + 0.1\log\frac{L_{\text{test}}}{L_{\text{train}}}\right)^2 \approx 1 + 0.2\log\frac{L_{\text{test}}}{L_{\text{train}}}. \tag{2}
$$

这与「熵不变性」推导的 $\log n$ Scale 形式相似，补偿了长度变化带来的影响。

## 与 NTK-RoPE 的关系

NTK-RoPE 通过修改 RoPE 底数实现外推：

$$
\theta_i = (10000\kappa)^{-2i/d}, \quad \kappa = \left(\frac{L_{\text{test}}}{L_{\text{train}}}\right)^{d/(d-2)}.
$$

YaRN 是对 [[kexue.fm/10122\|NTK-RoPE]] 的精细化改进——NTK 只让最后一个维度完整内插，而 YaRN 根据圈数对所有维度动态调整。

## 优势与应用

YaRN 的主要优势：

1. **实现简单**：仅修改 $\theta_i$，不改动模型结构
2. **零额外成本**：与 Flash Attention 等优化兼容
3. **即插即用**：无需重新训练

实验表明，YaRN 的外推效果在当时是「免训练」方法中的佼佼者。

## 小结

YaRN 通过「转圈视角」深化了对 RoPE 长度外推的理解：

1. **核心洞察**：关注单位圆上的点是否被充分训练，而非位置编号 OOD；
2. **解决思路**：根据圈数动态决定内插程度，实现高频外推、低频内插；
3. **实用价值**：简单、高效、即插即用。

至此，RoPE 系列的四节内容已完成：从基础理论（[[sec4_rope1\|第1节]]）到优化改进（[[sec4_rope2\|第2节]]），从变体探索（[[sec4_rope3\|第3节]]）到应用实践（[[sec4_rope4\|第4节]]）。RoPE 凭借其数学优雅性和实现高效性，已成为现代大模型位置编码的事实标准。

| [[sec4_rope3\|上一节]] | [[LLM/position embedding/index\|目录]] | —— |
| :-----------: | :----------------------------: | :--: |
