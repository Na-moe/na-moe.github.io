---
title: 第 2 节 绝对位置编码
---
| [[sec1_why_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec3_rel_pe\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |

在 RoPE 出现之前，出现了各种位置编码的尝试，主要可以分为绝对位置编码和相对位置编码。

本节我们主要介绍两类绝对位置编码：训练式和三角式；并且会对三角式进行简要的理论分析。

## 训练式绝对位置编码 —— Lrn.PE

训练式绝对位置编码就是为每个位置 $i$ 学习一个向量 $\boldsymbol{p}_{i}$, 然后将其与输入向量相加。

假设训练时的最大长度为 1024，那么就设置 1024 个维度为 $d$ 的可学习向量，将其与输入向量相加。

```python
def lrn_pe(
	seq_len: int,
	embedding_weight: Tensor['max_len', 'd'],
) -> Tensor['seq_len', 'd']:
    pos = torch.arange(seq_len, dtype=int)
    pe = F.embbeding(pos, embedding_weight)
    return pe
```

## 三角式绝对位置编码 —— Sin.PE

Google 在 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 中提出了三角式绝对位置编码。

其主要思想是使用正弦和余弦函数来表示位置编码：

$$
\begin{cases}
  \boldsymbol{p}_{i,2k} = \sin \left( \frac{i}{10000^{2k/d}} \right), \\
  \boldsymbol{p}_{i,2k+1} = \cos \left( \frac{i}{10000^{2k/d}} \right), 
\end{cases}
$$
  
其中 $\boldsymbol{p}_{i,2k}, \boldsymbol{p}_{i,2k+1}$ 分别是位置 $i$ 的第 $2k$ 和第 $2k+1$ 个维度的值。

```python
def sin_pe( # apply to x
	seq_len: int,
	d: int,
) -> Tensor['seq_len', d]:
	pos = torch.arange(seq_len, dtype=torch.float)[:, None]
	div_term = torch.exp(
		torch.arange(0, d, 2, dtype=float) *
		-(math.log(10000.0) / d)
	)
    pe = torch.zeros(seq_len, d_iodel)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe
```

我们通过泰勒展开推导三角式位置编码的合理性：

假设我们的模型为 $f(\cdots, \boldsymbol{x}_i, \cdots, \boldsymbol{x}_j, \cdots)$, 其中 $f$ 是标量函数。

对于不带因果掩码的模型，$f$ 是全对称的，即对于任意 $i,j$ 都有：

$$
f(\cdots, \boldsymbol{x}_i, \cdots, \boldsymbol{x}_j, \cdots) = f(\cdots, \boldsymbol{x}_j, \cdots, \boldsymbol{x}_i, \cdots)
$$

这也是其不能识别位置的主要原因。

而我们希望通过位置编码打破这种对称性：

$$
\tilde{f}(\cdots, \boldsymbol{x}_i, \cdots, \boldsymbol{x}_j, \cdots) = f(\cdots, \boldsymbol{x}_i + \boldsymbol{p}_i, \cdots, \boldsymbol{x}_j + \boldsymbol{p}_j, \cdots)
$$

为了简化，我们先只考虑 $i,j$ 这两个位置的情况，将位置编码视为扰动项，泰勒展开到二阶：

$$
  \tilde{f} = f + \boldsymbol{p}_i^\top \frac{\partial f}{\partial \boldsymbol{x}_i} + \boldsymbol{p}_j^\top \frac{\partial f}{\partial \boldsymbol{x}_j} + \frac{1}{2} \boldsymbol{p}_i^\top \frac{\partial^2 f}{\partial \boldsymbol{x}_i^2} + \frac{1}{2} \boldsymbol{p}_j^\top \frac{\partial^2 f}{\partial \boldsymbol{x}_j^2} + \underbrace{\boldsymbol{p}_i^\top \frac{\partial^2 f}{\partial \boldsymbol{x}_i \partial \boldsymbol{x}_j} \boldsymbol{p}_j}_{\boldsymbol{p}_i^\top \boldsymbol{\mathcal{H}} \boldsymbol{p}_j}
$$

可以看到：
* 第 1 项与位置无关；
* 而第 2 到 5 项都只和单一位置相关，因此是绝对位置信息；
* 而第 6 项是同时包含 $\boldsymbol{p}_i, \boldsymbol{p}_j$ 的交互项，我们希望它可以表达一定的相对位置信息。

### Sin.PE 的简化分析

我们先从简单的情形入手，假设 $\boldsymbol{\mathcal{H}} = \boldsymbol{I}$ 是单位阵。

此时 $\boldsymbol{p}_j^\top \boldsymbol{\mathcal{H}} \boldsymbol{p}_j = \boldsymbol{p}_j^\top \boldsymbol{p}_j = \langle \boldsymbol{p}_j, \boldsymbol{p}_j \rangle$ 是两个位置编码的内积。

我们希望这个内积可以表达两个位置的相对位置关系。

即存在某个函数 $g$，使得：

$$
\langle \boldsymbol{p}_i, \boldsymbol{p}_j \rangle = g(i-j)
$$

其中 $\boldsymbol{p}_i, \boldsymbol{p}_j$ 是 $d$ 维向量，我们先从 $d=2$ 的情形入手：

对于二维向量，我们将其视为复数，即将 $(x, y)$ 视为复数 $x+ \mathrm{i}y$，因此有：

$$
\langle \boldsymbol{p}_i, \boldsymbol{p}_j \rangle = \mathfrak{Re}\!\left[ \boldsymbol{p}_i \boldsymbol{p}_j^* \right]
$$

其中 $\boldsymbol{p}_j^*$ 是 $\boldsymbol{p}_j$ 的共轭复数。

为了满足 $\langle \boldsymbol{p}_i, \boldsymbol{p}_j \rangle = g(i-j)$，我们可以假设存在复数 $\boldsymbol{q}_{i-j}$ 使得：

$$
\boldsymbol{p}_i \boldsymbol{p}_j^* = \boldsymbol{q}_{i-j}
$$

我们利用复数的指数形式来表示，即设

$$
\begin{align}
  \boldsymbol{p}_i &= r_i\mathrm{e}^{\mathrm{i} \phi_i}, \\
  \boldsymbol{p}_j^* &= r_j\mathrm{e}^{-\mathrm{i} \phi_j}, \\
  \boldsymbol{q}_{i-j} &= t_{i-j}\mathrm{e}^{\mathrm{i} \psi_{i-j}},
\end{align}
$$

则有：

$$
\begin{aligned}
  &r_i r_j\mathrm{e}^{\mathrm{i} (\phi_j - \phi_j)} = t_{i-j}\mathrm{e}^{\mathrm{i} \psi_{i-j}} \\
  &\Rightarrow 
  \begin{cases}
    r_i r_j = t_{i-j}, \\
    \phi_i - \phi_j = \psi_{i-j},
  \end{cases}
\end{aligned}
$$

* 对于第一个方程，代入 $j=i$ 得 $r_i^2 = t_0$, 因此 $t_0$ 是一个常数
	* 可以设 $\boxed{r_{i}=t_0=1}$；
* 对于第二个方程，代入 $j = 0$ 得 $\phi_i - \phi_0 = \psi_i$, 简单起见，可以设 $\phi_0=0$；
	* 因此 $\psi_i = \phi_i$, 代入 $j=i-1$ 得 $\phi_i - \phi_{i-1} = \psi_1$；
	* 因此 $\{\phi_i\}$ 是一个等差数列，通解为 $\boxed{\phi_{i}= i \theta}$； 
 
所以二维情形下位置编码的解为：

$$
\boldsymbol{p}_i =\mathrm{e}^{\mathrm{i} i \theta}
 \Rightarrow 
\boldsymbol{p}_i = \begin{pmatrix}
  \ \cos i \theta \  \\
  \ \sin i \theta \ 
\end{pmatrix}
$$

而由于内积是线性叠加的，所以更高维的偶数维情形下位置编码可以表示为多个二维位置编码的组合：

$$
\boldsymbol{p}_j = \begin{pmatrix}
  \ \mathrm{e}^{\mathrm{i} i \theta_1} \ \\
  \ \mathrm{e}^{\mathrm{i} i \theta_2} \ \\
  \ \vdots \\
  \ \mathrm{e}^{\mathrm{i} i \theta_{d/2}} \ \\
\end{pmatrix}
\Rightarrow
\boldsymbol{p}_j = \begin{pmatrix}
  \ \cos i \theta_{1} \ \\
  \ \sin i \theta_{1} \ \\
  \ \cos i \theta_{2} \ \\
  \ \sin i \theta_{2} \ \\
  \ \vdots \\
  \ \cos i \theta_{d/2} \ \\
  \ \sin i \theta_{d/2} \ \\
\end{pmatrix}

$$

这个形式和论文中的形式已经非常接近了，只是 $\sin, \cos$ 位置稍有不同。

但是对于神经网络来说，神经元都是无序的，因此维度被打乱也是合理的位置编码。

而论文中对于 $\theta_k$ 的选择是 $\theta_k = 10000^{-(2k) / d}$, 我们将考察这个选择的性质。

### Sin.PE 的远程衰减

选择 $\theta_k = 10000^{-(2k) / d}$ 有一个良好的性质：随着 $\lvert i-j \rvert$ 的增大，$\langle \boldsymbol{p}_i, \boldsymbol{p}_j \rangle$ 有趋于零的趋势。

按照直觉，相对距离越大的输入之间的相关性应该越弱，这个性质符合我们的直觉，让我们分析一下这个性质的来源：

$$
\begin{aligned}
\langle \boldsymbol{p}_i, \boldsymbol{p}_j \rangle &= \mathfrak{Re}\!\left[ \mathrm{e}^{\mathrm{i}(i-j)\theta_1} + \mathrm{e}^{\mathrm{i}(i-j)\theta_2} + \dots + \mathrm{e}^{\mathrm{i}(i-j)\theta_{d / 2}} \right]  \\
&= \frac{d}{2} \mathfrak{Re}\!\left[ \sum_{i=1}^{d / 2} \mathrm{e}^{\mathrm{i}(i-j)10000^{-i / (d/2)}} \frac{1}{d/2} \right]  \\
&\approx \frac{d}{2} \mathfrak{Re}\!\left[ \int_0^1 \mathrm{e}^{\mathrm{i}(i-j)10000^{-t}} \, \mathrm{d}t \right] 
\end{aligned}
$$

我们通过数值计算的方式观察一下这个积分的结果：

![[sin_pe_attenuation.png|500]]

可以看到确实随着相对距离的增大而衰减。

### Sin.PE 的一般情况

当 $\boldsymbol{\mathcal{H}}$ 不是单位阵时，上述性质还能保留吗？

如果 $\boldsymbol{\mathcal{H}}$ 是一个对角阵，上述性质有一定程度的保留，此时有：

$$
\begin{aligned}
\boldsymbol{p}_i^\top \boldsymbol{\mathcal{H}} \boldsymbol{p}_j
&= \sum_{i=1}^{d / 2} \mathcal{H}_{2i, 2i} \cos i \theta_i \cos j \theta_i + \mathcal{H}_{2i+1, 2i+1} \sin i \theta_i \sin j \theta_i \\
&= \sum_{i=1}^{d / 2} \frac{1}{2} (\mathcal{H}_{2i, 2i} + \mathcal{H}_{2i+1, 2i+1}) \cos(i-j) \theta_i \\
& \quad + \sum_{i=1}^{d / 2} \frac{1}{2} (\mathcal{H}_{2i, 2i} - \mathcal{H}_{2i+1, 2i+1}) \cos(m+n) \theta_i
\end{aligned}
$$

也包含了相对位置 $i-j$, 但是多出来 $i+j$ 这一项，后者可以通过令 $\mathcal{H}_{2i,2i} = \mathcal{H}_{2i+1, 2i+1}$ 来消去。

而只考虑第一项，远程衰减则仍然存在：

$$
\sum_{i=1}^{d / 2} \frac{1}{2} (\mathcal{H}_{2i, 2i} + \mathcal{H}_{2i+1, 2i+1}) \cos(i-j) \theta_i \sim \int_0^1 h_t \mathrm{e}^{\mathrm{i} (i-j) \theta_t} \, \mathrm{d}t
$$

该积分在容易满足的条件下，可以有 $\lvert i-j \rvert \to \infty$ 时积分值趋于0的性质。

而如果 $\boldsymbol{\mathcal{H}}$ 不是对角阵，则这些性质很难得到，我们只能寄希望于 $\boldsymbol{\mathcal{H}}$ 的对角线部分占了主导，这样上述性质可以近似保留。而这意味着嵌入向量任意维度之间相关性较小，这在直觉上是可以满足的。

---

| [[sec1_why_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec3_rel_pe\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |

