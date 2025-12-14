---
title: 第 1 节 为什么需要位置编码？
---
| ——  | [[LLM/position embedding/index\|目录]] | [[sec2_abs_pe\|下一节]] |
| :-: | :----------------------------------: | :------------------: |

本节我们先讨论引入位置编码的主要原因——打破置换不变性；

接着，我们将说明，现代 Decoder-only LLMs 虽未直接使用位置编码，但其因果注意力机制隐式地起到了位置编码的作用；

最后，我们讨论了这种隐式位置编码的缺点。

## 置换不变性

Bert 时代的语言模型主要是基于双向注意力，输入的顺序并不会影响模型的输出。

为了让模型能够感知输入的顺序，我们需要给输入添加位置编码。

简单来说，位置编码最根本的作用是打破 Attention 的置换不变性。

什么是置换不变性呢？Bert 等模型的双向 Attention 基本形式为：

$$
\boldsymbol{y}_{i} = f(\boldsymbol{q}_{i}; \boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{L}) = \frac{{\sum_{j \leq L} \mathrm{e}^{ \boldsymbol{q}_{i} \cdot \boldsymbol{k}_{j}} } \boldsymbol{v}_{j}}{\sum_{j \leq L} \mathrm{e}^{ \boldsymbol{q}_{i} \cdot \boldsymbol{k}_{j}} },
$$


其中 $\boldsymbol{ \square }_{i} = \boldsymbol{x}_{i} \boldsymbol{W}_{\square}, \square \in \left\{ q,k,v \right\}$.

我们任取 ${1, 2, ..., L}$ 的一个排列 ${\sigma_{1},\sigma_{2}, \cdots, \sigma_{L}}$, 都有：

$$
\boldsymbol{y}_{i} = f(\boldsymbol{q}_{i}; \boldsymbol{x}_{\sigma_{1}}, \cdots, \boldsymbol{x}_{\sigma_{L}}) = f(\boldsymbol{q}_{i}; \boldsymbol{x}_{1}, \cdots , \boldsymbol{x}_{L})
$$

这就是置换不变性：我们交换 $\boldsymbol{x}_{j}$ 的顺序后 $\boldsymbol{y}_{i}$ 的值并不会改变。

这和自然语言的性质相悖。位置编码的引入就是为了打破这个置换不变性。

## 最简单的位置编码 —— NoPE

事实上，在 LLM 时代，不添加位置编码（即 NoPE）也可以起到打破置换不变性的效果。

现在的 LLM 都是 Causal Attention，形式如下：

$$
\boldsymbol{y}_{i} = f(\boldsymbol{q}_{i}; \boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{L}) = \frac{{\sum_{j \leq i} \mathrm{e}^{ \boldsymbol{q_{i} \cdot \boldsymbol{k}_{j}} } \boldsymbol{v}_{j}}}{\sum_{j \leq i} \mathrm{e}^{ \boldsymbol{q}_{i} \cdot \boldsymbol{k}_{j}} }.
$$

相比双向 Attention, 求和上限从序列长度 $L$ 变为了当前 token 所在的位置 $i$.

这使得 $\boldsymbol{y}_{1}, \cdots, \boldsymbol{y}_{L}$ 的结果依赖于 $\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{L}$ 的顺序。

### Causal NoPE 编码位置到模长

进一步地，我们分析一下 Causal Attention 是通过什么机制来实现位置编码的效果的。

我们知道 Causal Attention 相当于对 $\left\{ \boldsymbol{v}_{i} \right\}$ 的加权求和。

先尝试最简单的加权形式，即均匀加权，也即考虑这样的权重矩阵：

$$
\begin{pmatrix}
\ 1 & & &\ \\
\ \frac{1}{2} & \frac{1}{2} & &\ \\
\ \vdots & \vdots & \ddots &\ \\
\ \frac{1}{L} & \frac{1}{L} & \cdots & \frac{1}{L}\ \\
\end{pmatrix}
$$

有：

$$
\boldsymbol{y}_{i} = \frac{1}{i}\sum_{j\leq i} \boldsymbol{v}_{j}.
$$

假设 $\boldsymbol{v}_{j}$ 是 $\mathrm{i.i.d} \sim \mathcal{N}\left( 0, \sigma^{2} \right)$ ，那么就有：

$$
\frac{1}{d}\sum_{k=1}^{d}{y_{i,k}} \approx \mathbb{E}\!\left[ y_{i,k} \right] = \mathbb{E}\! \left[ \frac{1}{i} \sum_{j\leq i} v_{j,k} \right] = \frac{1}{i} \sum_{j\leq i} \mathbb{E}\!\left[ v_{j,k} \right] = 0,  
$$

$$
\frac{1}{d}\sum_{k=1}^{d}{y_{i,k}^2} \approx \mathbb{E}\!\left[ y_{i,k}^2 \right] = \mathbb{E}\! \left[ \frac{1}{i} \sum_{j\leq i} v_{j,k}^2 \right] = \frac{1}{i} \sum_{j\leq i} \mathbb{E}\!\left[ v_{j,k}^2 \right] = \frac{\sigma^2}{i},  
$$

$$
\lVert \boldsymbol{y}_{i} \rVert = \sqrt{ \frac{1}{d}\sum_{k=1}^{d}{y_{i,k}^2} } \approx \frac{\sigma}{\sqrt{ i }}
$$

也就是说，*使用 NoPE 时，$\boldsymbol{y}_{i}$ 的位置信息其实通过 $\frac{1}{\sqrt{ i }}$ 的系数藏在了 $\ell_2$  范数里。*

### NoPE 有何不足

主要有以下几点：

1. 相当于只用了一个标量 $\frac{1}{\sqrt{ i }}$ 进行位置编码，效果有限；
2. $i$ 很大时，$\frac{1}{i}$ 和 $\frac{1}{i+1}$ 的区分度很差；
3. 只建模了绝对位置，没有相对位置的感知能力。

---

| ——  | [[LLM/position embedding/index\|目录]] | [[sec2_abs_pe\|下一节]] |
| :-: | :----------------------------------: | :------------------: |
