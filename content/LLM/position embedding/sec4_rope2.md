---
title: 第 4 节 旋转位置编码-2
---
| [[sec3_rel_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope1\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |

## 第一性原理推导 RoPE

首先明确我们的目标：
1. 给 $\boldsymbol{q}, \boldsymbol{k}$ 添加上绝对位置编码；
2. 使 $\boldsymbol{q}\boldsymbol{k}^\top$ 的结果带有相对位置信息。

对于目标 1，我们假设通过函数 $\boldsymbol{f}: \mathbb{R}^{d} \times [\![1, L]\!] \mapsto \mathbb{R}^{d}$ 编码绝对位置，即：

$$
\tilde{\boldsymbol{q}}_{i} = \boldsymbol{f}\!\left( \boldsymbol{q}, i \right), \  \tilde{\boldsymbol{k}}_{j} = \boldsymbol{f}\!\left( \boldsymbol{k}, j \right).
$$

为了达成目标 2，我们希望 $\langle \tilde{\boldsymbol{q}}_{i}, \tilde{\boldsymbol{k}}_{j} \rangle = g\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right)$, 其中 $g$ 是标量函数。

可以假设一些初始条件帮助求解，设 $\boldsymbol{f}\!\left( \boldsymbol{q}, 0 \right) = \boldsymbol{q}, \boldsymbol{f}\!\left( \boldsymbol{k}, 0 \right)=\boldsymbol{k}$.

类似于 [[sec2_abs_pe#Sin.PE 的简化分析|Sin.PE]] 中的操作，先考虑二维情形，我们有：

$$
\mathfrak{Re}\!\left[ \boldsymbol{f}\!\left( \boldsymbol{q}, i \right) \boldsymbol{f}^*\!\left( \boldsymbol{k}, j \right) \right] = g\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right).
$$

不妨设存在复数 $\boldsymbol{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right)$ 满足 $\boldsymbol{f}\!\left( \boldsymbol{q}, i \right) \boldsymbol{f}^*\!\left( \boldsymbol{k}, j \right) = \boldsymbol{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right)$, 指数形式表示为：

$$
\begin{aligned}
\boldsymbol{f}\!\left( \boldsymbol{q}, i \right) &= R_{f}\!\left( \boldsymbol{q}, i \right) \mathrm{e}^{ \mathrm{i} \Theta_{f}\left( \boldsymbol{q}, i \right)  } \\
\boldsymbol{f}\!\left( \boldsymbol{k}, j \right) &= R_{f}\!\left( \boldsymbol{k}, j \right)\mathrm{e}^{ \mathrm{i} \Theta_{f}\left( \boldsymbol{k}, j \right)  } \\
\boldsymbol{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right) &= R_{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right)\mathrm{e}^{ \mathrm{i} \Theta_{g}\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right)  }
\end{aligned}
$$

可得方程组：

$$
\begin{cases}
  R_{f}\!\left( \boldsymbol{q}, i \right) R_{f}\!\left( \boldsymbol{k}, j \right) = R_{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right) \\
  \Theta_{f}\!\left( \boldsymbol{q}, i \right) - \Theta_{f}\!\left( \boldsymbol{k}, j \right) = \Theta_{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, i-j \right)
\end{cases}
$$

* 对于第一个方程，带入 $j=i$ 得到：
	* $R_{f}\!\left( \boldsymbol{q}, i \right) R_{f}\!\left( \boldsymbol{k}, i \right) = R_{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, 0 \right) = R_{f}\!\left( \boldsymbol{q}, 0 \right) R_{f}\!\left( \boldsymbol{k}, 0 \right) = \lVert \boldsymbol{q} \rVert \lVert \boldsymbol{k} \rVert$
	* 因此可以设 $\boxed{R_{f}\!\left( \boldsymbol{q}, i \right) = \lVert \boldsymbol{q} \rVert, R_{f}\!\left( \boldsymbol{k}, j \right) = \lVert \boldsymbol{k} \rVert}$, 即模长不依赖于位置；
* 对于第二个方程，同样带入 $j=i$ 得到：
	* $\Theta_{f}\!\left( \boldsymbol{q}, i \right) - \Theta_{f}\!\left( \boldsymbol{k}, i \right) = \Theta_{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, 0 \right) = \Theta(\boldsymbol{q}) - \Theta \!\left( \boldsymbol{k} \right)$, 其中 $\Theta \!\left( \boldsymbol{q} \right), \Theta \!\left( \boldsymbol{k} \right)$ 是 $\boldsymbol{q}, \boldsymbol{k}$ 本身的角度；
	* 整理得 $\Theta_{f}\!\left( \boldsymbol{q}, i \right) - \Theta \!\left( \boldsymbol{q} \right) = \Theta_{f}\!\left( \boldsymbol{k}, i \right) - \Theta \!\left( \boldsymbol{k} \right)$, 所以 $\Theta_{f}\!\left( \boldsymbol{q}, i \right) - \Theta \!\left( \boldsymbol{q} \right)$ 应该是一个和 $i$ 有关，和 $\boldsymbol{q}$ 无关的函数，记为 $\varphi\!\left( i \right)$；
* 对于第二个方程，再带入 $j=i-1$, 整理得到：
	* $\varphi \!\left( i \right) - \varphi \!\left( i-1 \right) = \underbrace{ \Theta_{g}\!\left( \boldsymbol{q}, \boldsymbol{k}, 1 \right) + \Theta \!\left( \boldsymbol{q} \right) - \Theta \!\left( \boldsymbol{k} \right) }_{ \theta }$；
	* 所以 $\left\{ \varphi \!\left( i \right) \right\}$ 是等差数列，通解为 $\varphi \!\left( i \right) = i\theta$；
	* 因此 $\boxed{\Theta_{f}\!\left( \boldsymbol{q},  i \right) = \Theta \!\left( \boldsymbol{q} \right) + i\theta}$.

综上，我们得到 $\boldsymbol{f}$：

$$
\begin{aligned}
  \boldsymbol{f}\!\left( \boldsymbol{q}, i \right) &= \lVert \boldsymbol{q} \rVert \mathrm{e}^{ \mathrm{i} \left( \Theta \left( \boldsymbol{q} \right) + i\theta \right)  }  =\boldsymbol{q}\mathrm{e}^{ \mathrm{i}i\theta } \\
  &= \begin{pmatrix}
    \ \cos i\theta & -\sin i\theta \ \\
    \ \sin i\theta & \cos i\theta
  \end{pmatrix} \begin{pmatrix}
    \ q_{1} \ \\
    \ q_{2} \
  \end{pmatrix}
\end{aligned}

$$

由于内积线性可加，因此可以扩展到任意偶数维度：

$$
\boldsymbol{f}\!\left( \boldsymbol{q}, i \right) = \underbrace{ \begin{pmatrix}
  \ \cos i\theta_{1} & -\sin i\theta_{1} & 0 & 0 & \cdots  & 0 & 0 \ \\
  \ \sin i\theta_{1} & \cos i\theta_{1} & 0 & 0 & \cdots  & 0 & 0 \ \\
  \ 0 & 0 & \cos i\theta_{2} & -\sin i\theta_{2} & \cdots  & 0 & 0 \ \\
  \ 0 & 0 & \sin i\theta_{2} & \cos i\theta_{2} & \cdots  & 0 & 0 \ \\ 
  \ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \ \\
  \ 0 & 0 & 0 & 0 & \cdots  & \cos i\theta_{\frac{d}{2}} & -\sin i\theta_{\frac{d}{2}} \ \\
  \ 0 & 0 & 0 & 0 & \cdots  & \cos i\theta_{\frac{d}{2}} & -\sin i\theta_{\frac{d}{2}} \ \\
\end{pmatrix} }_{ \boldsymbol{\mathcal{R}}_{i} }
\begin{pmatrix}
  \ q_{1} \ \\
  \ q_{2} \ \\
  \ q_{3} \ \\
  \ q_{4} \ \\
  \ \vdots \ \\
  \ q_{d-1} \ \\
  \ q_{d} \
\end{pmatrix}
$$

值得指出的是，$\boldsymbol{\mathcal{R}}_{i}$ 是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

一个高效的实现方式是：

$$

\begin{pmatrix}
  \ q_{1} \ \\
  \ q_{2} \ \\
  \ q_{3} \ \\
  \ q_{4} \ \\
  \ \vdots \ \\
  \ q_{d-1} \ \\
  \ q_{d} \
\end{pmatrix} \otimes
\begin{pmatrix}
  \ \cos i\theta_{1} \ \\
  \ \cos i\theta_{1} \ \\
  \ \cos i\theta_{2} \ \\
  \ \cos i\theta_{2} \ \\
  \ \vdots \ \\
  \ \cos i\theta_{\frac{d}{2}} \ \\
  \ \cos i\theta_{\frac{d}{2}} \
\end{pmatrix} +
\begin{pmatrix}
  \ q_{2} \ \\
  \ -q_{1} \ \\
  \ q_{4} \ \\
  \ -q_{3} \ \\
  \ \vdots \ \\
  \ q_{d} \ \\
  \ -q_{d-1} \
\end{pmatrix} \otimes
\begin{pmatrix}
  \ \sin i\theta_{1} \ \\
  \ \sin i\theta_{1} \ \\
  \ \sin i\theta_{2} \ \\
  \ \sin i\theta_{2} \ \\
  \ \vdots \ \\
  \ \sin i\theta_{\frac{d}{2}} \ \\
  \ \sin i\theta_{\frac{d}{2}} \
\end{pmatrix}
$$

```python
def rope(
	x: Tensor['seq_len', 'd'],
	freq: float = 10000.0,
) -> Tensor['seq_len', 'd']:
	seq_len, d = x.shape
	
	pos = torch.arange(
		seq_len, dtype=float, device=x.device
	)[:, None]
	theta = torch.exp(
		math.log(freq) * -torch.arange(0, d, 2)/d
	)[None, :]
	
	cos = torch.cos(pos * theta)
	sin = torch.sin(pos * theta)
	
	evens = x[:, 0::2]
	odds  = x[:, 1::2]
	
	pe = evens * cos + odds * cos - evens * sin + odd * cos
	return pe
```

## RoPE 的远程衰减

$(\boldsymbol{\mathcal{R}}_{i}\boldsymbol{q}_{i})^\top(\boldsymbol{\mathcal{R}}_{j}\boldsymbol{k}_{j})$ 的结果同样具有远程衰减性。

证明如下：

$$
(\boldsymbol{\mathcal{R}}_{i}\boldsymbol{q}_{i})^\top(\boldsymbol{\mathcal{R}}_{j}\boldsymbol{k}_{j}) = \mathfrak{Re}\!\left[ \sum_{k=1}^{d/2} \boldsymbol{q}_{\left[ 2k-1:2k \right] } \boldsymbol{k}^*_{\left[ 2k-1:2k \right] } \mathrm{e}^{ \mathrm{i} \left( i-j \right)\theta_{k}  } \right] 
$$

记 $h_{k}=\boldsymbol{q}_{\left[ 2k-1:2k \right] } \boldsymbol{k}^*_{\left[ 2k-1:2k \right]}, S_{n}=\sum_{k=1}^{n-1} \mathrm{e}^{ \mathrm{i}\left( i-j \right)\theta_{k} }$, 约定 $h_{d/2+1}=0, S_{0}=0$, 由 Abel 分部求和法：

$$
\sum_{k=1}^{d/2} \boldsymbol{q}_{\left[ 2k-1:2k \right] } \boldsymbol{k}^*_{\left[ 2k-1:2k \right] } \mathrm{e}^{ \mathrm{i} \left( i-j \right)\theta_{k} } = \sum_{k=1}^{d/2} h_{k} \left( S_{k+1} - S_{k} \right) = - \sum_{k=1}^{d/2} S_{k+1} \left( h_{k+1} - h_{k} \right) 
$$

所以：

$$
\begin{aligned}
  \left| \sum_{k=1}^{d/2} \boldsymbol{q}_{\left[ 2k-1:2k \right] } \boldsymbol{k}^*_{\left[ 2k-1:2k \right] } \mathrm{e}^{ \mathrm{i} \left( i-j \right)\theta_{k} } \right| &= \left|  \sum_{k=1}^{d/2} S_{k+1} \left( h_{k+1} - h_{k} \right)  \right| \\
  &\leq \sum_{k=1}^{d/2} \left| S_{k+1} \right| \left| h_{k+1} - h_{k} \right| \\
  &\leq \left( {\max_{k} \left| h_{k+1} - h_{k} \right|} \right)  \sum_{k=1}^{d/2} \left| S_{k+1} \right|
\end{aligned}
$$

因此可以通过考察 $\sum_{k=1}^{d/2} \left| S_{k+1} \right|=\sum_{k=1}^{d/2}\left| \sum_{m=1}^k \mathrm{e}^{ \mathrm{i} \left( i-j \right) \theta_{m} } \right|$ 的随着相对距离的变化情况来作为原式远程衰减性的体现，绘图如下：

![[rope_attenuation.png]]


| [[sec3_rel_pe\|上一节]] | [[LLM/position embedding/index\|目录]] | [[sec4_rope1\|下一节]] |
| :-----------: | :----------------------------: | :----------------: |
