---
title: 第 5 章 核方法
---
| [[chapter4_generative_learning_algorithms\|上一章]] | [[CS229_CN/index#目录\|目录]] | [[chapter6_support_vector_machines\|下一章]] |
| :----------------------------------------------: | :--------------------: | :---------------------------------------: |

## 5.1 特征映射

回顾在线性回归的讨论中，考虑了根据房屋的居住面积 (记为 $x$ ) 预测房屋价格 (记为 $y$ ) 的问题，并将 $x$ 的线性函数拟合到训练数据。如果价格 $y$ 可以更准确地表示为 $x$ 的 *非线性 (non-linear)* 函数呢？在这种情况下，需要一个比线性模型更具表现力的模型族。

首先考虑拟合三次函数 $y = \theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0$。结果表明，可以将三次函数视为在不同特征变量集（定义如下）上的线性函数。具体来说，令函数 $\phi: \mathbb{R} \to \mathbb{R}^4$ 定义为

^eq5-1
$$
\begin{equation}
    \phi(x) = 
	    \begin{bmatrix} 
		    \ 1 \ \\
		    \ x \ \\
		    \ x^2 \ \\
		    \ x^3 \ 
	    \end{bmatrix} 
	    \in \mathbb{R}^4. \tag{5.1}
\end{equation}
$$

令 $\theta \in \mathbb{R}^4$ 是包含 $\theta_0, \theta_1, \theta_2, \theta_3$ 作为元素的向量。那么可以将三次函数写成 $x$ 的形式：

$$
\theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0 = \theta^T \phi(x).
$$

因此，变量 $x$ 的三次函数可以视为变量 $\phi(x)$ 上的线性函数。为了区分这两组变量，在核方法的背景下，将问题的「原始」输入值称为输入 **属性 (attributes)** (在本例中为 $x$，即居住面积)。当原始输入被映射到一组新的量 $\phi(x)$ 时，将这些新的量称为 **特征 (features)** 变量。(不幸的是，不同的作者在不同的语境下使用不同的术语来描述这两者。) 将 $\phi$ 称为 **特征映射 (feature map)**，它将属性映射到特征。

## 5.2 特征的最小均方

现在我们推导拟合模型 $\theta^T \phi(x)$ 的梯度下降算法。首先回顾一下，对于普通的最小二乘问题，拟合 $\theta^T x$ 的批量梯度下降更新（其推导参见讲义第一章）为：

^eq5-2
$$
\begin{align} 
    \theta &:= \theta + \alpha \sum_{i=1}^n (y^{(i)} - h_\theta(x^{(i)})) x^{(i)} \notag\\ 
    &:= \theta + \alpha \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)}) x^{(i)}. \tag{5.2}
\end{align}
$$

令 $\phi: \mathbb{R}^d \to \mathbb{R}^p$ 是一个将属性 $x$ (在 $\mathbb{R}^d$ 中) 映射到 $\mathbb{R}^p$ 中的特征 $\phi(x)$ 的特征映射。(在前面小节的示例中，$d=1$ 且 $p=4$.) 现在目标是拟合函数 $\theta^T \phi(x)$, 其中 $\theta$ 是 $\mathbb{R}^p$ 中的向量而不是 $\mathbb{R}^d$ 中的向量。可以将上面算法中 $x^{(i)}$ 的所有出现替换为 $\phi(x^{(i)})$, 得到新的更新：

^eq5-3
$$
\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)})) \phi(x^{(i)}). \tag{5.3}
$$

类似地，相应的随机梯度下降更新规则是

$$
\theta := \theta + \alpha (y^{(i)} - \theta^T \phi(x^{(i)})) \phi(x^{(i)}). \tag{5.4}
$$

## 5.3 使用核技巧的最小均方

当特征 $\phi(x)$ 是高维向量时，上面的梯度下降或随机梯度下降更新在计算上变得昂贵。举例来说，考虑将方程 [[chapter5_kernel_methods#^eq5-1|(5.1)]] 中的特征映射直接扩展到高维输入 $x$: 假设 $x \in \mathbb{R}^d$, 令 $\phi(x)$ 是包含所有 $x$ 的次数 $\le 3$ 的项的向量

^eq5-5
$$
\phi(x) = 
	\begin{bmatrix} 
		\ 1 \ \\
		\ x_1 \ \\
		\ x_2 \ \\
		\ \vdots \ \\
		\ x_1^2 \ \\
		\ x_1 x_2 \ \\
		\ x_1 x_3 \ \\
		\ \vdots \ \\
		\ x_2 x_1 \ \\
		\ \vdots \ \\
		\ x_1^3 \ \\
		\ x_1^2 x_2 \ \\
		\ \vdots \ \\
		\ x_2 x_3 x_1 \ \\
		\ \vdots 
	\end{bmatrix}. \tag{5.5}
$$

特征 $\phi(x)$ 的维度大约是 $d^3$ 量级。[^1] 对于计算来说，这是一个非常长的向量——当 $d = 1000$ 时，每次更新需要至少计算和存储一个 $1000^3 = 10^9$ 维向量，这比普通最小二乘更新规则 [[chapter5_kernel_methods#^eq5-2|(5.2)]] 慢 $10^6$ 倍。

乍一看，每次更新 $d^3$ 的运行时长和内存使用似乎是不可避免的，因为向量 $\theta$ 本身的维度是 $p \approx d^3$, 并且需要更新和存储 $\theta$ 的每一个元素。然而，将引入核技巧，更新它不需要显式存储 $\theta$，并且可以显著改善运行时长。

为简单起见，假设初始值 $\theta = 0$, 并且关注迭代更新 [[chapter5_kernel_methods#^eq5-3|(5.3)]]。主要观察是，在任何时候，$\theta$ 都可以表示为向量 $\phi(x^{(1)}), \dots, \phi(x^{(n)})$ 的线性组合。实际上，可以通过归纳法证明如下。在初始化时，$\theta = 0 = \sum_{i=1}^n 0 \cdot \phi(x^{(i)})$. 假设在某个时刻，$\theta$ 可以表示为

$$
\theta = \sum_{i=1}^n \beta_i \phi(x^{(i)}). \tag{5.6}
$$

其中 $\beta_1, \dots, \beta_n \in \mathbb{R}$. 然后，断言在下一轮中，$\theta$ 仍然是 $\phi(x^{(1)}), \dots, \phi(x^{(n)})$ 的线性组合，因为

$$
\begin{align} 
    \theta 
	    &:= \theta + \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)})) \phi(x^{(i)}) \notag\\ 
	    &= \sum_{i=1}^n \beta_i \phi(x^{(i)}) + \alpha \sum_{i=1}^n (y^{(i)} - \theta^T \phi(x^{(i)})) \phi(x^{(i)}) \notag\\ 
	    &= \sum_{i=1}^n \underbrace{(\beta_i + \alpha (y^{(i)} - \theta^T \phi(x^{(i)})))}_{\text{新的}\  \beta_i} \phi(x^{(i)}). \tag{5.7}
\end{align}
$$

可以意识到，一般策略是通过一组系数 $\beta_1, \dots, \beta_n$ 隐式表示 $p$ 维向量 $\theta$. 为了做到这一点，推导系数 $\beta_i$ 的更新规则。使用上面的方程，可以看到新的 $\beta_i$ 依赖于旧的 $\beta_i$

$$
\beta_i := \beta_i + \alpha (y^{(i)} - \theta^T \phi(x^{(i)})). \tag{5.8}
$$

这里，方程的右侧仍然有旧的 $\theta$。将 $\theta$ 替换为 $\theta = \sum_{j=1}^n \beta_j \phi(x^{(j)})$ 得到

$$
\forall i \in \{1, \dots, n\}, \beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^n \beta_j \phi(x^{(j)})^T \phi(x^{(i)}) \right).
$$

通常将 $\phi(x^{(j)})^T \phi(x^{(i)})$ 重写为 $\langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle$ 以强调它是两个特征向量的内积。通过将 $\beta_i$ 视为 $\theta$ 的新表示，已经成功地将批梯度下降算法转化为一个迭代更新 $\beta$ 值的新算法。它可能看起来在每次迭代时，仍然需要计算所有 $i, j$ 对的 $\langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle$ 的值，其中每个计算可能需要大约 $O(p)$ 的操作。然而，有两个重要的特性可以解决这个问题：

1. 在循环开始之前，可以预先计算所有 $i, j$ 对的成对内积 $\langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle$.
2. 对于定义在 [[chapter5_kernel_methods#^eq5-5|(5.5)]] 中的特征映射 $\phi$ (或许多其他有趣的特征映射)，计算 $\langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle$ 可以是高效的并且不必需显式计算 $\phi(x^{(i)})$. 这是因为：

^eq5-9
$$
\begin{align} 
	\langle \phi(x), \phi(z) \rangle 
		&= 1 + \sum_{i=1}^d x_i z_i + \sum_{i,j \in \{1, \dots, d\}} x_i x_j z_i z_j + \sum_{i,j,k \in \{1, \dots, d\}} x_i x_j x_k z_i z_j z_k \notag\\ 
		&= 1 + \sum_{i=1}^d x_i z_i + \left( \sum_{i=1}^d x_i z_i \right)^2 + \left( \sum_{i=1}^d x_i z_i \right)^3 \notag\\ 
		&= 1 + \langle x, z \rangle + \langle x, z \rangle^2 + \langle x, z \rangle^3 \tag{5.9}
\end{align}
$$
- 因此，要计算 $\langle \phi(x), \phi(z) \rangle$, 可以首先用 $O(d)$ 的时间计算 $\langle x, z \rangle$, 然后进行一些常数次操作来计算 $1 + \langle x, z \rangle + \langle x, z \rangle^2 + \langle x, z \rangle^3$.

正如将看到的，特征 $\phi(x), \phi(z)$ 之间的内积在这里至关重要。将与特征映射 $\phi$ 对应的 **核 (Kernel)** 定义为一个 $\mathcal{X} \times \mathcal{X} \to \mathbb{R}$ 的函数，满足：[^2]

$$
K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle
$$

最终算法总结如下：

---
1. 用方程 [[chapter5_kernel_methods#^eq5-9|(5.9)]] 对所有 $i, j \in \{1, \dots, n\}$ 计算 $K(x^{(i)}, x^{(j)}) \triangleq \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle$. 设置 $\beta := 0$.
2. **循环：**

^eq5-11
$$
\forall i \in \{1, \dots, n\}, \beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^n \beta_j K(x^{(i)}, x^{(j)}) \right).
        \tag{5.11}
$$
* 或者用向量表示法，令 $K$ 是一个 $n \times n$ 矩阵，其中 $K_{ij} = K(x^{(i)}, x^{(j)})$, 有

$$
\beta := \beta + \alpha (\vec{y} - K \beta)
$$
---

通过上面的算法，可以在每次更新时以 $O(n)$ 的时间高效地更新向量 $\theta$ 的表示 $\beta$. 最后，需要证明表示 $\beta$ 的知识足以计算预测 $\theta^T \phi(x)$. 实际上，有

^eq5-12
$$
\theta^T \phi(x) = \sum_{i=1}^n \beta_i \phi(x^{(i)})^T \phi(x) = \sum_{i=1}^n \beta_i K(x^{(i)}, x). \tag{5.12}
$$

可以意识到，本质上所有需要知道的关于特征映射 $\phi(\cdot)$ 的信息都封装在相应的核函数 $K(\cdot, \cdot)$ 中。将在下一节中对此进行详细阐述。

## 5.4 核的性质

在上一小节中，从一个显式定义的特征映射 $\phi$ 开始，导出了核函数 $K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle$. 然后，看到核函数是如此本质，只要核函数被定义，整个训练算法就可以完全用核方法的语言编写，而无需引用特征映射 $\phi$, 因此对于测试示例 $x$ 的预测 (方程[[chapter5_kernel_methods#^eq5-12|(5.12)]]) 也是如此。

因此，可以尝试定义其他核函数 $K(\cdot, \cdot)$ 并运行算法 [[chapter5_kernel_methods#^eq5-11|(5.11)]]。请注意，算法 [[chapter5_kernel_methods#^eq5-11|(5.11)]] 不需要显式访问特征映射 $\phi$, 因此只需要确保特征映射 $\phi$ 的存在，但不一定需要能够显式写下 $\phi$.

哪些类型的函数 $K(\cdot, \cdot)$ 可以对应于某个特征映射 $\phi$? 换句话说，能否判断出来是否存在某个特征映射 $\phi$ 使得对于所有 $x, z$ 都有 $K(x, z) = \phi(x)^T \phi(z)$?

如果可以通过给出有效核函数的精确表征来回答这个问题，那么就可以完全改变选择核函数 $K$ 的接口，而不是选择特征映射 $\phi$ 的接口。具体来说，可以选取一个函数 $K$，验证它满足该表征 (从而存在一个与 $K$ 对应的特征映射 $\phi$ )，然后就可以运行更新规则 [[chapter5_kernel_methods#^eq5-11|(5.11)]]。这里的好处是，不需要能够计算或解析地写下 $\phi$, 只需要知道它的存在性。在本小节的末尾，将通过几个具体的核示例回答这个问题。

假设 $x, z \in \mathbb{R}^d$, 首先考虑函数 $K(\cdot, \cdot)$ 定义为：

$$
K(x, z) = (x^T z)^2.
$$

也可以将其写为

$$
\begin{align} 
    K(x, z) 
	    &= \left( \sum_{i=1}^d x_i z_i \right) \left( \sum_{j=1}^d x_j z_j \right) \\
	    &= \sum_{i=1}^d \sum_{j=1}^d x_i x_j z_i z_j \\
	    &= \sum_{i,j=1}^d (x_i x_j)(z_i z_j) 
\end{align}
$$

因此，可以看到 $K(x, z) = \langle \phi(x), \phi(z) \rangle$ 是与特征映射 $\phi$ 对应的核函数 (这里以 $d=3$ 的情况为例) 由下式给出

$$
\phi(x) = \begin{bmatrix}
        \ x_1 x_1\ \\
        \ x_1 x_2\ \\
        \ x_1 x_3\ \\
        \ x_2 x_1\ \\
        \ x_2 x_2\ \\
        \ x_2 x_3\ \\
        \ x_3 x_1\ \\
        \ x_3 x_2\ \\
        \ x_3 x_3\
    \end{bmatrix}.
$$

请回想一下核的计算效率，注意，虽然计算高维的 $\phi(x)$ 需要 $O(d^2)$ 的时间，但找到 $K(x, z)$ 只需 $O(d)$ 的时间——与输入属性的维度呈线性关系。

对于另一个相关的例子，也考虑由下式定义的 $K(\cdot, \cdot)$：

$$
\begin{align}
K(x, z) 
	&= (x^T z + c)^2 \\
	&= \sum_{i,j=1}^d (x_i x_j)(z_i z_j) + \sum_{i=1}^d (\sqrt{2c} x_i)(\sqrt{2c} z_i) + c^2.
\end{align}
$$

(请自行验证) 这个函数 $K$ 是一个核函数，它对应到特征映射 (再次以 $d=3$ 为例)

$$
\phi(x) = \begin{bmatrix}
        \ x_1 x_1\ \\
        \ x_1 x_2\ \\
        \ x_1 x_3\ \\
        \ x_2 x_1\ \\
        \ x_2 x_2\ \\
        \ x_2 x_3\ \\
        \ x_3 x_1\ \\
        \ x_3 x_2\ \\
        \ x_3 x_3\ \\
        \ \sqrt{2c} x_1\ \\
        \ \sqrt{2c} x_2\ \\
        \ \sqrt{2c} x_3\ \\
        \ c\
    \end{bmatrix},
$$

其中参数 $c$ 控制 $x_i$ (一阶) 项和 $x_i x_j$ (二阶) 项之间的相对权重。

更广泛地说，核 $K(x, z) = (x^T z + c)^k$ 对应于一个特征空间，包含所有次数不超过 $k$ 的一元多项式 $x_{i_1} x_{i_2} \dots x_{i_k}$. 然而尽管在这个 $O(d^k)$ 维的高维空间中工作，计算 $K(x, z)$ 仍然只需要 $O(d)$ 的时间，因此即使在如此高维的特征空间中，也不需要显式表示出特征向量。

### 核作为相似性度量

现在，讨论一下核的另一种视角。直观地 (尽管这种直觉有一些问题，但先忽略)，如果 $\phi(x)$ 和 $\phi(z)$ 彼此接近，那么可以期望 $K(x, z) = \phi(x)^T \phi(z)$ 很大。反过来，如果 $\phi(x)$ 和 $\phi(z)$ 相距很远——例如，几乎相互正交——那么 $K(x, z) = \phi(x)^T \phi(z)$ 将很小。因此，可以将 $K(x, z)$ 看作是 $\phi(x)$ 和 $\phi(z)$ 之间相似性的一种度量，或者说是 $x$ 和 $z$ 之间相似性的一种度量。

有了这种直觉，假设对于正在研究的某个学习问题，想出了一个函数 $K(x, z)$, 认为它可能是衡量 $x$ 和 $z$ 相似性的合理度量。例如，可能选择了

$$
K(x, z) = \exp \left( -\frac{\|x - z\|^2}{2\sigma^2} \right).
$$

这是衡量 $x$ 和 $z$ 相似性的合理度量，当 $x$ 和 $z$ 接近时接近 $1$, 当 $x$ 和 $z$ 相距很远时接近 $0$. 是否存在一个特征映射 $\phi$ 使得上面定义的核 $K$ 满足 $K(x, z) = \phi(x)^T \phi(z)$? 在这个特殊的例子中，答案是肯定的。这个核被称为 **高斯核 (Gaussian kernel)**，并且对应于一个无限维的特征映射 $\phi$. 下面，我们将精确地描述一个函数 $K$ 需要满足哪些性质才能成为一个有效的核函数，即对应于某个特征映射 $\phi$.

### 有效核的必要条件

现在假设 $K$ 确实是一个有效的核，对应于某个特征映射 $\phi$，首先看看它满足哪些性质。考虑一个包含 $n$ 个点 (不一定是训练集) 的有限集合 $\{x^{(1)}, \dots, x^{(n)}\}$, 并定义一个 $n \times n$ 的方阵 $K$, 其 $(i, j)$ 项由 $K_{ij} = K(x^{(i)}, x^{(j)})$ 给出。这个矩阵称为 **核矩阵 (kernel matrix)**。请注意，我们为了方便使用了 $K$ 来同时表示核函数 $K(x, z)$ 和核矩阵 $K$, 因为它们之间有明显的密切关系。

如果 $K$ 是一个有效的核，那么有 $K_{ij} = K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)}) = \phi(x^{(j)})^T \phi(x^{(i)}) = K(x^{(j)}, x^{(i)}) = K_{ji}$，因此 $K$ 必须是对称的。此外，令 $\phi_k(x)$ 表示向量 $\phi(x)$ 的第 $k$ 个坐标，对于任意向量 $z$，有

$$
\begin{aligned} 
    z^T K z 
	    &= \sum_i \sum_j z_i K_{ij} z_j \\ 
	    &= \sum_i \sum_j z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j \\ 
	    &= \sum_i \sum_j z_i \left( \sum_k \phi_k(x^{(i)}) \phi_k(x^{(j)}) \right) z_j \\ 
	    &= \sum_k \sum_i \sum_j z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\ 
	    &= \sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right) \left( \sum_j z_j \phi_k(x^{(j)}) \right) \\ 
	    &= \sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right)^2 \\ 
	    &\ge 0. 
\end{aligned}
$$

倒数第二步利用了 $\sum_{i,j} a_i a_j = (\sum_i a_i)^2$, 其中 $a_i = z_i \phi_k(x^{(i)})$. 由于 $z$ 是任意的，这表明 $K$ 是半正定的 ($K \ge 0$)。

因此，证明出，如果 $K$ 是一个有效的核 (即，它对应于某个特征映射 $\phi$ )，那么相应的核矩阵 $K \in \mathbb{R}^{n \times n}$ 是对称半正定的。

### 有效核的充分条件

更一般地，上面的条件不仅是必要的，而且也是 $K$ 成为有效核 (也称为 Mercer 核) 的充分条件。以下结果归功于 Mercer。[^3]

**定理 (Mercer).** 对于给定的 $K: \mathbb{R}^d \times \mathbb{R}^d \mapsto \mathbb{R}$, 则 $K$ 是一个有效核的充要条件是，对于任意有限集合 $\{x^{(1)}, \dots, x^{(n)}\}$ ($n < \infty$), 相应的核矩阵是对称半正定的。

给定一个函数 $K$，除了尝试找到一个与之对应的特征映射 $\phi$ 之外，这个定理提供了另一种测试它是否是有效核的方法。在问题集 2 中也将有机会进一步探索这些想法。

课上还简要讨论了几个其他核的例子。例如，考虑手写数字识别问题，给定一个手写数字 (0-9) 的图像 (16x16 像素)，需要确定它是哪个数字。使用简单的多项式核 $K(x, z) = (x^T z)^k$ 或高斯核，SVM 能够在此问题上获得非常好的性能。这尤其令人惊讶，因为输入属性 $x$ 仅仅是图像像素值的 256 维向量，系统没有关于视觉的先验知识，甚至没有关于哪些像素相邻的知识。在课堂上简要讨论的另一个例子是，如果试图对字符串进行分类 (例如，$x$ 是由氨基酸串成的蛋白质)，那么构建一个合理的、「小」的特征集对于大多数学习算法来说似乎很困难，特别是如果不同的字符串长度不同。然而，考虑令 $\phi(x)$ 是一个特征向量，它计算 $x$ 中每个长度为 $k$ 的子字符串的出现次数。如果考虑英文字母组成的字符串，那么有 $26^k$ 个这样的字符串。因此，$\phi(x)$ 是一个 $26^k$ 维向量；即使对于中等大小的 $k$, 这可能也太大了，无法有效处理 (例如，$26^4 \approx 460000$)。但是，使用 (类似动态规划的) 字符串匹配算法，可以有效地计算 $K(x, z) = \phi(x)^T \phi(z)$, 这样就可以在这个 $26^k$ 维特征空间中隐式地工作，而无需显式计算特征向量。

### 核方法的应用

我们已经看到了核方法在线性回归中的应用。在下一部分，将介绍支持向量机，核方法可以直接应用于其中。这里不再赘述。实际上，核方法的思想比线性回归和 SVM 具有更广泛的适用性。具体来说，如果有一个学习算法，可以完全用输入属性向量之间的内积 $\langle x, z \rangle$ 来表达，那么通过将其替换为核 $K(x, z)$ (其中 $K$ 是一个核)，就可以「神奇地」让算法在对应于 $K$ 的高维特征空间中高效工作。例如，这个核技巧可以应用于感知机，推导出核感知机算法。本课程后面将看到的许多算法也将适用于这种方法，这种方法被称为「核技巧」。

| [[chapter4_generative_learning_algorithms\|上一章]] | [[CS229_CN/index#目录\|目录]] | [[chapter6_support_vector_machines\|下一章]] |
| :----------------------------------------------: | :--------------------: | :---------------------------------------: |

[^1]: 此处，为简单起见，包含所有重复的单项式 (因此，$x_1 x_2 x_3$ 和 $x_2 x_3 x_1$ 都出现在 $\phi(x)$ 中)。所以，$\phi(x)$ 中共有 $1 + d + d^2 + d^3$ 个元素。

[^2]: 回想一下， $\mathcal{X}$ 是输入 $x$ 的取值空间。在当前示例中，$\mathcal{X} = \mathbb{R}^d$.

[^3]: 许多文献给出 Mercer 定理时，涉及到 $L^2$ 函数的稍复杂形式，但当输入属性取值为 $\mathbb{R}^d$ 时，我们这里给出的版本是等价的。
