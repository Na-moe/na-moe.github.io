---
title: 第 4 节 旋转位置编码-3
---

| [[sec4_rope2\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope4\|下一节]] |
| :-----------------: | :----------------------------------: | :----------------------: |

在前两节中，我们探讨了在 Query 和 Key 上施加旋转的 [[sec4_rope1\|QK-RoPE]]，以及它的改进版本 [[sec4_rope2\|Partial-RoPE]]。一个自然的问题是：**RoPE 可以加在 Value 上吗？** 本节我们将介绍 **VO-RoPE**（Value-Output RoPE），通过 Value 和 Output 的双重旋转实现相对位置编码，为理解 RoPE 提供新的视角。

## VO-RoPE 的构造

### 直接加在 V 上的问题

回顾 [[sec3_rel_pe\|Attention]] 的基本形式（参见 [[sec4_rope1\|RoPE 一节]] 的背景）：

$$
\boldsymbol{o}_{i} = \sum_{j} a_{i,j}\boldsymbol{v}_{j}, \quad a_{i,j} = \text{softmax}(\boldsymbol{q}_{i}^{\top}\boldsymbol{k}_{j}).
$$

若直接将 RoPE 加在 $\boldsymbol{v}_{j}$ 上：

$$
\boldsymbol{o}_{i} = \sum_{j} a_{i,j} \boldsymbol{\mathcal{R}}_{j}\boldsymbol{v}_{j},
$$

输出显式依赖于绝对位置 $j$，失去了相对位置编码的特性。

### 逆向旋转恢复相对位置

解决方法是给输出再加一次**逆向**旋转：

^eq1
$$
\boldsymbol{o}_{i} = \boldsymbol{\mathcal{R}}_{i}^{\top}\left(\sum_{j} a_{i,j} \boldsymbol{\mathcal{R}}_{j}\boldsymbol{v}_{j}\right). \tag{1}
$$

利用 $\boldsymbol{\mathcal{R}}_{i}^{\top}\boldsymbol{\mathcal{R}}_{j} = \boldsymbol{\mathcal{R}}_{j-i}$：

$$
\boldsymbol{o}_{i} = \sum_{j} a_{i,j} \boldsymbol{\mathcal{R}}_{j-i}\boldsymbol{v}_{j}.
$$

输出再次变成**相对位置编码**的形式！这种在 Value 和 Output 上施加 RoPE 的方法称为 **VO-RoPE**（或「第二类旋转位置编码」），以区别于 [[sec4_rope1\|第一类的 QK-RoPE]]。

## 实验对比

在约 1B 参数的类 LLaMA 模型上测试：

| 位置编码 | Loss | 说明 |
| :------- | :--: | :--- |
| **QK-RoPE** | **2.712** | 标准 RoPE |
| QKVO-RoPE | 2.719 | QK + VO 叠加 |
| K-RoPE | 2.769 | 仅在 Key 上加 |
| *VO-RoPE* | *2.770* | 仅在 VO 上加 |
| NoPE | 2.795 | 无位置编码 |

**关键发现**：
1. QK-RoPE $\approx$ QKVO-RoPE：叠加 VO-RoPE 无明显增益；
2. VO-RoPE $>$ NoPE：VO-RoPE 确实有效；
3. VO-RoPE $<$ QK-RoPE：在当前设置下，QK-RoPE 仍是最佳选择。

## 与 MLA 的结合

尽管 VO-RoPE 在标准 Attention 中优势不明显，但它在 **MLA** 中具有独特价值。

### MLA 的困境

MLA 推理阶段可视为 K、V 共享的 MQA：

$$
\boldsymbol{o}_{i} = \sum_{j} a_{i,j}\boldsymbol{c}_{j}, \quad \boldsymbol{c}_{j} = \boldsymbol{x}_j\boldsymbol{W}_c.
$$

若用 [[sec4_rope1\|QK-RoPE]]：
- Value 不加 RoPE → K、V 不完全共享，KV Cache 翻倍；
- Value 加 RoPE → 失去相对位置特性。

### VO-RoPE 的解决方案

利用 [[sec4_rope3\|VO-RoPE]]（回想 [[sec4_rope1#相对位置的内积\|第一节]] 中旋转矩阵的性质），可以在保持 K、V 共享的同时实现相对位置编码：

^eq2
$$
\boldsymbol{o}_{i} = \boldsymbol{\mathcal{R}}_{i}^{\top}\sum_{j} a_{i,j}(\boldsymbol{\mathcal{R}}_{j}\boldsymbol{c}_{j}), \quad a_{i,j} = (\boldsymbol{\mathcal{R}}_{i}\boldsymbol{q}_{i})^{\top} (\boldsymbol{\mathcal{R}}_{j}\boldsymbol{c}_{j}). \tag{2}
$$

这样，$\boldsymbol{c}$ 既作为 Key 又作为 Value（经旋转后），保持了单一 KV Cache 的优势，同时通过输出端的逆向旋转恢复了相对位置特性。

## 与线性 RNN 的联系

VO-RoPE 还提供了从 Attention 到**复线性 RNN**的过渡形式。考虑因果场景，取 $a_{i,j} = \gamma^{i-j}$（指数衰减）：

$$
\boldsymbol{o}_{i} = \sum_{j=1}^{i} \gamma^{i-j} \boldsymbol{\mathcal{R}}_{j-i}\boldsymbol{v}_{j} = \sum_{j=1}^{i} (\gamma e^{-\mathbb{I}\theta})^{i-j} \boldsymbol{v}_{j}.
$$

这正是**带复数衰减的线性 RNN**！VO-RoPE 因此在理论上连接了 Attention 与线性 RNN（如 LRU、RetNet）。

## 小结

VO-RoPE 通过 Value 和 Output 的双重旋转实现了相对位置编码：

1. **构造巧妙**：通过逆向旋转恢复相对位置特性；
2. **效果稳健**：优于 NoPE，但略逊于 QK-RoPE；
3. **应用独特**：在 MLA 等 K、V 共享场景中具有不可替代的价值；
4. **理论意义**：连接 Attention 与复线性 RNN。

在 [[sec4_rope4\|下一节]] 中，我们将探讨 [[sec4_rope1\|RoPE]] 在实际应用中的一个重要问题——[[sec4_rope4\|长度外推]]，介绍 YaRN 这一基于「转圈视角」的高效方法。

| [[sec4_rope2\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope4\|下一节]] |
| :-----------------: | :----------------------------------: | :----------------------: |
