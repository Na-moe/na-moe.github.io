---
title: 位置编码
---
在位置编码的这系列博客中，我们将简要介绍位置编码的发展历史。

从为什么需要位置编码开始，到中间一系列位置编码的探索，最后到「现代位置编码」 RoPE。

之后我们会介绍一些基于 RoPE 的改进：

* 增强长文本扩展性的 YaRN；
* 在 VO 上也进行位置编码的 VO-RoPE；
* 只在部分维度上应用 RoPE 的 $\partial$-RoPE.

为了一致性，我们对符号做如下约定：

| 符号                                                           | 值域                                 | 描述                                                          |
| ------------------------------------------------------------ | ---------------------------------- | ----------------------------------------------------------- |
| $L$                                                          | $\mathbb{N}^{+}$                   | 序列长度                                                        |
| $i,j$                                                        | $[\![1,L]\!]$                      | 序列索引                                                        |
| $d$                                                          | $\mathbb{N}^{+}$                   | 模型的嵌入维度                                                     |
| $\boldsymbol{x}_{i}, \boldsymbol{y}_{i}$                     | $\mathbb{R}^{d}$                   | 第 $i$ 个输入/输出向量                                              |
| $\boldsymbol{q}_{i},\boldsymbol{k}_{i},\boldsymbol{v}_{i}$   | $\mathbb{R}^{d}$                   | 第 $i$ 个 $\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}$ 向量 |
| $\boldsymbol{W}_{q}, \boldsymbol{W}_{k}, \boldsymbol{W}_{v}$ | $\mathbb{R}^{d\times d}$           | $q,k,v$ 的参数矩阵                                               |
| $\boldsymbol{p}_{i}$                                         | $\mathbb{R}^{d}$                   | 位置 $i$ 的位置编码                                                |
| $\boldsymbol{R}_{i,j}$                                       | $\mathbb{R}^{d}$                   | 位置 $i$ 相对于位置 $j$ 的位置编码                                      |
| $\boldsymbol{W}_{R}$                                         | $\mathbb{R}^{d\times d}$           | $\boldsymbol{R}$ 的参数矩阵                                      |
| $\tilde{\mathbb{\Lambda}}(\cdot)$                            | $\mathbb{R}^{n} \mapsto (0,1)^{n}$ | $\mathrm{softmax}$ 函数                                       |
| $\mathfrak{Re}[\cdot]$                                       | $\mathbb{C} \mapsto \mathbb{R}$    | 取实部                                                         |

## 目录

> [!example]- [[LLM/position embedding/index|位置编码]]  
>   
> > [!example]- [[sec1_why_pe|第 1 节 为什么需要位置编码]]  
> > 
> >   &emsp;╠ [[sec1_why_pe#置换不变性|置换不变性]]  
> >   &emsp;╚ [[sec1_why_pe#最简单的位置编码 —— NoPE|最简单的位置编码 —— NoPE]]  
> >   &emsp;&emsp;&nbsp;╠ [[sec1_why_pe#Causal NoPE 编码位置到模长|Causal NoPE 编码位置到模长]]  
> >   &emsp;&emsp;&nbsp;╚ [[sec1_why_pe#NoPE 有何不足|NoPE 有何不足]]  
>   
> > [!example]-  [[sec2_abs_pe|第 2 节 绝对位置编码]]  
> > 
> >   &emsp;╠ [[sec2_abs_pe#训练式绝对位置编码 —— Lrn.PE|训练式绝对位置编码 —— Lrn.PE]]  
> >   &emsp;╚ [[sec2_abs_pe#三角式绝对位置编码 —— Sin.PE|三角式绝对位置编码 —— Sin.PE]]  
> >   &emsp;&emsp;&nbsp;╠ [[sec2_abs_pe#Sin.PE 的简化分析|Sin.PE 的简化分析]]  
> >   &emsp;&emsp;&nbsp;╠ [[sec2_abs_pe#Sin.PE 的远程衰减|Sin.PE 的远程衰减]]  
> >   &emsp;&emsp;&nbsp;╚ [[sec2_abs_pe#Sin.PE 的一般情况|Sin.PE 的一般情况]]   
>   
> > [!example]-  [[sec3_rel_pe|第 3 节 相对位置编码]]  
> > 
> >   &emsp;╠ [[sec3_rel_pe#经典相对位置编码 —— Rel.PE|经典相对位置编码 —— Rel.PE]]  
> >   &emsp;╠ [[sec3_rel_pe#相对位置编码变体 —— XLNet|相对位置编码变体 —— XLNet]]  
> >   &emsp;╠ [[sec3_rel_pe#相对位置编码变体 —— T5|相对位置编码变体 —— T5]]  
> >   &emsp;╚ [[sec3_rel_pe#相对位置编码变体 —— DeBERTa|相对位置编码变体 —— DeBERTa]]  
>   
> > [!example]-  [[sec4_rope1|第 4 节 旋转位置编码]]  
> > 
> >   &emsp;╠ [[sec4_rope1#第一性原理推导 RoPE|第一性原理推导 RoPE]]  
> >   &emsp;╚ [[sec4_rope1#RoPE 的远程衰减|RoPE 的远程衰减]]  
>   
> > [!example]-  [[chapter5_kernel_methods|第 5 章 核方法]]  
> > 
> >   &emsp;╠ [[chapter5_kernel_methods#5.1 特征映射|5.1 特征映射]]  
> >   &emsp;╠ [[chapter5_kernel_methods#5.2 特征的最小均方|5.2 特征的最小均方]]  
> >   &emsp;╠ [[chapter5_kernel_methods#5.3 使用核技巧的最小均方|5.3 使用核技巧的最小均方]]  
> >   &emsp;╚ [[chapter5_kernel_methods#5.4 核的性质|5.4 核的性质]]  
>   
> > [!example]-  [[chapter6_support_vector_machines|第 6 章 支持向量机]]  
> > 
> >   &emsp;╠ [[chapter6_support_vector_machines#6.1 间隔：直觉|6.1 间隔：直觉]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.2 符号 (选读)|6.2 符号 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.3 函数间隔与几何间隔 (选读)|6.3 函数间隔与几何间隔 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.4 最优间隔分类器 (选读)|6.4 最优间隔分类器 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.5 拉格朗日对偶 (选读)|6.5 拉格朗日对偶 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.6 最优间隔分类器：对偶形式 (选读)|6.6 最优间隔分类器：对偶形式 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.7 正则化与非线性可分情况 (选读)|6.7 正则化与非线性可分情况 (选读)]]  
> >   &emsp;╚ [[chapter6_support_vector_machines#6.8 SMO 算法 (选读)|6.8 SMO 算法 (选读)]]  

