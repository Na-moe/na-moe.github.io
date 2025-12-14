---
title: 第 2 节 绝对位置编码
---
| [[sec1_why_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec3_rel_pe\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |

在 RoPE 出现之前，出现了各种位置编码的尝试，主要可以分为绝对位置编码和相对位置编码。本节我们主要介绍两类绝对位置编码：训练式和三角式，并且会对三角式进行简要的理论分析。

## 训练式绝对位置编码 —— Lrn.PE

训练式绝对位置编码就是为每个位置 $n$ 学习一个向量 $\boldsymbol{p}_{n}$，然后将其和输入的嵌入向量相加。假设训练时的最大长度为 1024，嵌入维度为 768，那么就设置 1024 个维度为 768 的可学习向量，将其和输入的嵌入向量相加。

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

Google 在 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 中提出了三角式绝对位置编码。其主要思想是使用正弦和余弦函数来表示位置编码：

$$
\begin{cases}
  \boldsymbol{p}_{k,2i} = \sin \left( \frac{k}{10000^{2i/d}} \right), \\
  \boldsymbol{p}_{k,2i+1} = \cos \left( \frac{k}{10000^{2i/d}} \right), 
\end{cases}
$$
  
其中 $\boldsymbol{p}_{k,2i}, \boldsymbol{p}_{2i+1}$ 分别是位置 $k$ 的第 $2i$ 和第 $2i+1$ 个维度的值，$d$ 是嵌入向量的维度。

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
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe
```

### Sin.PE 的简化分析

### Sin.PE 的远程衰减

### Sin.PE 的一般情况

---

| [[sec1_why_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec3_rel_pe\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |

