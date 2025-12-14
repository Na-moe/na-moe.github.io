---
title: 第 3 节 相对位置编码
---
| [[sec2_abs_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope1\|下一节]] |
| :-----------: | ------------------------------ | ------------------ |

---

绝对位置编码的主要问题是，其只建模了绝对位置的关系，但是现实中自然语言更多关注词元之间相对位置的关系。因此，有许多相对位置编码被提出，本节将从 $\boldsymbol{q}_i \boldsymbol{k}_j^\top$ 的展开式开始，介绍一系列相对位置编码：Rel.PE、XLNet 式、T5 式和 DeBERTa 式。

**tl;dr:**

先考虑绝对位置编码的情况：

$$
\begin{cases}
  \boldsymbol{y}_i = \sum_j \tilde{\mathbb{\Lambda}}_j\big(\boldsymbol{q}_i \boldsymbol{k}_j^\top\big) \boldsymbol{v}_j \\
  \boldsymbol{\square}_{\triangle} = (\boldsymbol{x}_\triangle + \boldsymbol{p}_\triangle) \boldsymbol{W}_{\square},\ \text{其中 }\square \in \left\{ q,k,v \right\}, \triangle \in \left\{ i,j \right\}
\end{cases}
$$

展开 $\boldsymbol{q}_i \boldsymbol{k}_j^\top$ 得到：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^\top = \underbrace{ \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top }_{ \text{内容-内容} } +
  \underbrace{ \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{p}_j^\top }_{ \text{内容-位置} } +
  \underbrace{ \boldsymbol{p}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top }_{ \text{位置-内容} } +
  \underbrace{ \boldsymbol{p}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{p}_j^\top }_{ \text{位置-位置} }
$$

各类相对位置编码对上式的修改如下：

| 名称      | 方法                                                                                                                                          |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Rel.PE  | 保留前两项，用相对位置编码 $\boldsymbol{R}_{i,j}$ 替代 $\boldsymbol{p}_j\boldsymbol{W}_k$                                                                  |
| XLNet   | $\boldsymbol{p}_{j}$ 替换为 $\boldsymbol{R}_{i,j}$, $\boldsymbol{p}_{i}\boldsymbol{W}_{q}$ 替换为可学习参数 $\boldsymbol{\theta},\boldsymbol{\varphi}$ |
| T5      | 保留一、四项，用可学习偏置 $\boldsymbol{\beta}_{i,j}$ 替代第四项                                                                                              |
| DeBERTa | 保留前三项，用相对位置编码 $\boldsymbol{R}_{i,j}, \boldsymbol{R}_{j,i}$ 替代 $\boldsymbol{p}_i, \boldsymbol{p}_{j}$                                        |

## 经典相对位置编码 —— Rel.PE

相对位置编码起源于 Google 的论文《Self-Attention with Relative Position epresentations》，一般认为是从绝对位置编码中启发得到的：

$$
\begin{cases}
  \boldsymbol{y}_i = \sum_j \mathrm{softmax}_j\big(\boldsymbol{q}_i \boldsymbol{k}_j^\top\big) \boldsymbol{v}_j \\
  \boldsymbol{q}_i = (\boldsymbol{x}_i + \boldsymbol{p}_i) \boldsymbol{W}_q \\
  \boldsymbol{k}_j = (\boldsymbol{x}_j + \boldsymbol{p}_j) \boldsymbol{W}_k \\
  \boldsymbol{v}_j = (\boldsymbol{x}_j + \boldsymbol{p}_j) \boldsymbol{W}_v
\end{cases}
$$

我们初步展开 $\boldsymbol{q}_i \boldsymbol{k}_j^\top$：

$$
\begin{aligned}
  \boldsymbol{q}_i \boldsymbol{k}_j^\top 
    &= (\boldsymbol{x}_i + \boldsymbol{p}_i) \boldsymbol{W}_q \boldsymbol{W}_k^\top (\boldsymbol{x}_j + \boldsymbol{p}_j)^\top \\
    &= (\boldsymbol{x}_i \boldsymbol{W}_q + \boldsymbol{p}_i \boldsymbol{W}_q)(\boldsymbol{x}_j \boldsymbol{W}_k + \boldsymbol{p}_j \boldsymbol{W}_k)^\top
\end{aligned}
$$

为了引入相对位置信息，可以把第一项位置去掉，第二项 $\boldsymbol{p}_j \boldsymbol{W}_k$ 改成二元位置向量 $\boldsymbol{R}_{i,j}^K$：

$$
\boldsymbol{q}_i \boldsymbol{k}_j^\top = \boldsymbol{x}_i \boldsymbol{W}_q (\boldsymbol{x}_j \boldsymbol{W}_k + \boldsymbol{R}_{i,j}^K)^\top
$$

然后将 $\boldsymbol{v}_j = (\boldsymbol{x}_j + \boldsymbol{p}_j) \boldsymbol{W}_v = \boldsymbol{x}_j \boldsymbol{W}_v + \boldsymbol{p}_j \boldsymbol{W}_v$ 中的 $\boldsymbol{p}_j \boldsymbol{W}_v$ 换成 $\boldsymbol{R}_{i,j}^V$：

$$
\boldsymbol{y}_i = \sum_j \tilde{\boldsymbol{\Lambda}}_j\Big(\boldsymbol{x}_i \boldsymbol{W}_q (\boldsymbol{x}_j \boldsymbol{W}_k + \boldsymbol{R}_{i,j}^K)^\top\Big) (\boldsymbol{x}_j \boldsymbol{W}_v + \boldsymbol{R}_{i,j}^V)
$$

而相对位置 $\boldsymbol{R}_{i,j}^V$ 可以只依赖于相对位置 $j - i$，因此可以用一个绝对位置编码 $\boldsymbol{p}_{i-j}$ 来表示。具体形式可以选择训练式和三角式任一。

```python
def rel_pe( # apply to x
	seq_len: int,
	cur_pos: int,
	clip_min: int = 0,
	clip_max: int = 1024,
) -> Tensor['seq_len', 'd']:
	pos = torch.arange(seq_len, dtype=float)
	rel_pos = cur_pos - pos
	rel_pos[rel_pos>clip_max] = clip_max
	rel_pos[rel_pos<clip_min] = clip_min
	pe = lrn_pe(rel_pos) # can also be sin_pe
	return pe
```

## 相对位置编码变体 —— XLNet

XLNet 式位置编码来源自 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)，不过更广为人知则是到 [XLNet]("https://arxiv.org/abs/1906.08237") 模型超过Bert之后，因此被称为 XLNet 式位置编码。

XLNet 式位置编码源于对于 $\boldsymbol{q}_i \boldsymbol{k}_j^\top$ 的完全展开：

$$
\begin{aligned}
  \boldsymbol{q}_i \boldsymbol{k}_j^\top
  = \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top +
  \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{p}_j^\top +
  \boldsymbol{p}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top +
  \boldsymbol{p}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{p}_j^\top
\end{aligned}
$$

其做法很简单，是直接将 $\boldsymbol{p}_j$ 替换为相对位置向量 $\boldsymbol{R}_{i-j}^\top$，而 $\boldsymbol{p}_i$ 则直接替换为两个可训练的向量 $\boldsymbol{\theta},\boldsymbol{\varphi}$：

$$

  \boldsymbol{q}_i \boldsymbol{k}_j^\top
  = \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top +
  \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{R}_{i-j}^\top +
  \boldsymbol{\theta} \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top +
  \boldsymbol{\varphi} \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{R}_{i-j}^\top

$$

由于选择的绝对位置编码中 $\boldsymbol{R}_{i-j}$ 的空间和 $\boldsymbol{x}_j$ 未必相同，所以需要将 $\boldsymbol{W}_k$ 换成另外的独立矩阵 $\boldsymbol{W}_R$，并且 $\boldsymbol{\theta} \boldsymbol{W}_q, \boldsymbol{\varphi} \boldsymbol{W}_q$ 可以直接合并作 $\boldsymbol{\theta}, \boldsymbol{\varphi}$，因此最终得到：

$$

  \boldsymbol{q}_i \boldsymbol{k}_j^\top

  = \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top +

  \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_R^\top \boldsymbol{R}_{i-j}^\top +

  \boldsymbol{\theta} \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top +

  \boldsymbol{\varphi} \boldsymbol{W}_R^\top \boldsymbol{R}_{i-j}^\top

$$

此外，$\boldsymbol{v}_j$ 上的位置编码直接被去掉了，似乎之后的工作很少在 $\boldsymbol{v}_j$ 上添加位置编码了。

```python
def xlnet_pe( # apply to q@k.T
	cur_pos: int,
	seq_len: int,
	q_i: Tensor['dim'], 
	k: Tensor['seq_len', 'dim'],
	W_r: Tensor['dim', 'dim'],
	theta: Tensor['dim'],
	phi: Tensor['dim'],
) -> Tensor['seq_len']:
	r = rel_pe(seq_len, cur_pos) @ W_r
	pe = q_i @ r.T + theta @ k.T + phi @ r.T
	return pe
```

## 相对位置编码变体 —— T5

T5 式位置编码来源自 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)。思路和 XLNet 类似，但是进行了更进一步的分析： $\boldsymbol{q}_i \boldsymbol{k}_j^\top$ 的完全展开每一项的含义可以理解为：“输入-输入”，“输入-位置”，“位置-输入”，“位置-位置”四种关系。如果认为输入信息和位置信息是解耦的，那么它们就不该有太多的交互，因此可以直接删去，并且最后一项实际上是一个只依赖于位置 $(i,j)$ 的标量，因此可以直接训练得到：

$$
  \boldsymbol{q}_i \boldsymbol{k}_j^\top = \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top + \boldsymbol{\beta}_{i, j}
$$

相对于只是在 Attention 矩阵上增加一个可训练的偏置 $\boldsymbol{\beta} \in \mathbb{R}^{L \times L}$。

```python
def t5_pe( # apply to q@k.T
	seq_len: int,
	cur_pos: int,
	beta: Tensor['max_len', 'max_len'],
) -> Tensor['seq_len']:
	pos = torch.arange(seq_len, dtype=int)
	pe = beta[cur_pos, pos]
	return pe
```

## 相对位置编码变体 —— DeBERTa

DeBERTa 式位置编码来源自 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)。其思路和 T5 类似，但是是去掉了第 4 项，而保留 2、3 项：

$$

  \boldsymbol{q}_i \boldsymbol{k}_j^\top = \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top +
  \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{R}_{i, j}^\top +
  \boldsymbol{R}_{j, i} \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_j^\top
$$

也取得了不错的效果。

```python
def deberta_pe(
	seq_len: int,
	cur_pos: int,
	q_i: Tensor['dim'],
	k: Tensor['seq_len', 'dim'],
	W_q: Tensor['dim', 'dim'],
	W_k: Tensor['dim', 'dim'],
) -> Tensor['seq_len']:
	r = rel_pe(seq_len, cur_pos)
	pe = q_i @ (r @ W_k).T + (r @ W_q) @ k.T
	return pe
```

---

| [[sec2_abs_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope1\|下一节]] |
| :-----------: | ------------------------------ | ------------------ |