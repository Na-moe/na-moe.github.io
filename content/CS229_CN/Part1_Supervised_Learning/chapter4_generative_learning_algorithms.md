---
title: 第 4 章 生成式学习算法
---

| [[chapter3_generalized_linear_model\|上一章]] | [[CS229_CN/index\|目录]] | 下一章 |
| :----------------------------------------: | :--------------------: | :-: |

到目前为止，我们讨论的主要是学习这样的一类算法，其建模给定 $x$ 的情况下 $y$ 的条件分布 $p(y|x; \theta)$. 例如，逻辑回归将 $p(y|x; \theta)$ 建模为 $h_\theta(x) = g(\theta^T x)$, 其中 $g$ 是 $\text{sigmoid}$ 函数。在本章中，将讨论一种不同类型的学习算法。

考虑一个分类问题，其中希望根据动物的一些特征来区分大象 ($y=1$) 和狗 ($y=0$). 给定一个训练集，像逻辑回归或感知机算法 (本质上) 试图找到一条直线——也就是一个决策边界——来分隔大象和狗。然后，为了将新动物分类为大象或狗，检查其落在决策边界的哪一侧，并据此做出预测。

这里介绍一种不同的方法。首先观察大象，可以建立一个关于大象外观的模型。然后观察狗，可以建立一个关于狗外观的独立模型。最后，为了对新动物进行分类，可以将新动物与大象模型进行匹配，并将其与狗模型进行匹配，以查看新动物是否更像在训练集中看到的大象或狗。

试图直接学习 $p(y|x)$ 的算法 (如逻辑回归)，或试图直接学习从输入空间 $\mathcal{X}$ 到标签 $\{0, 1\}$ 的映射的算法 (如感知机算法) 称为 **判别式 (discriminative)** 学习算法。这里，将讨论那些试图建模 $p(x|y)$ (和 $p(y)$) 的算法。这些算法称为 **生成式 (generative)** 学习算法。例如，如果 $y$ 表示一个样本是狗 ($0$) 还是大象 ($1$), 则 $p(x|y=0)$ 建模狗的特征分布，$p(x|y=1)$ 建模大象的特征分布。

在建模 $p(y)$ (称为 **类先验 (class priors)**) 和 $p(x|y)$ 之后，可以利用贝叶斯定理推导出给定 $x$ 时 $y$ 的后验分布：

$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}.
$$

这里，分母由 $p(x) = p(x|y=1)p(y=1) + p(x|y=0)p(y=0)$ 给出 (可以根据概率的标准性质来验证这一点)，因此也可以用学到的 $p(x|y)$ 和 $p(y)$ 来表示。实际上，如果计算 $p(y|x)$ 是为了进行预测，那么并不需要计算分母，因为

$$
\begin{aligned}
    \arg \max_y p(y|x) 
	    &= \arg \max_y \frac{p(x|y)p(y)}{p(x)} \\
	    &= \arg \max_y p(x|y)p(y).
\end{aligned}
$$

## 4.1 高斯判别分析

将要介绍的第一个生成式学习算法是高斯判别分析 (GDA)。在这个模型中，假设 $p(x|y)$ 服从多元正态分布。在介绍 GDA 模型本身之前，先简要讨论一下多元正态分布的性质。

### 4.1.1 多元正态分布

$d$ 维的多元正态分布，也称为多元高斯分布，由 **均值向量 (mean vector)** $\mu \in \mathbb{R}^d$ 和 **协方差矩阵 (covariance matrix)** $\Sigma \in \mathbb{R}^{d \times d}$ 参数化，其中 $\Sigma \ge 0$ 是对称半正定矩阵。其密度函数写为 $\mathcal{N}(\mu, \Sigma)$，形式如下：

$$
p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right).
$$

在上面的方程中，$|\Sigma|$ 表示矩阵 $\Sigma$ 的行列式。
对于服从 $\mathcal{N}(\mu, \Sigma)$ 分布的随机变量 $X$，其均值（毫不意外地）由 $\mu$ 给出：

$$
\mathrm{E}[X] = \int_x x p(x; \mu, \Sigma) dx = \mu
$$

向量值随机变量 $Z$ 的 **协方差 (covariance)** 定义为 $\text{Cov}(Z) = \mathrm{E}[(Z - \mathrm{E}[Z])(Z - \mathrm{E}[Z])^T]$. 这推广了实值随机变量的方差概念。协方差也可以定义为 $\text{Cov}(Z) = \mathrm{E}[ZZ^T] - (\mathrm{E}[Z])(\mathrm{E}[Z])^T$. (可以自行证明这两个定义是等价的。) 如果 $X \sim \mathcal{N}(\mu, \Sigma)$, 则

$$
\mathrm{Cov}(X) = \Sigma.
$$

以下是一些高斯分布密度函数的示例：

![[gaussian_scale.svg]]

最左边的图显示的是均值为零（即 $2 \times 1$ 零向量）、协方差矩阵为 $\Sigma = I$（即 $2 \times 2$ 单位矩阵）的高斯分布。均值为零、协方差为单位矩阵的高斯分布也称为 **标准正态分布 (standard normal distribution)**。中间的图显示的是均值为零、$\Sigma = 0.6I$ 的高斯分布的密度；最右边的图显示的是 $\Sigma = 2I$ 的高斯分布的密度。可以看到，随着 $\Sigma$ 变大，高斯分布变得更加“分散”，而随着 $\Sigma$ 变小，分布变得更加“紧凑”。

接下来再看一些例子。

![[gaussian_cov.svg]]

上面的图显示了均值为 0、协方差矩阵分别为

$$
\Sigma = \begin{bmatrix} \ 1& 0\ \\ \ 0& 1\ \end{bmatrix}; \ 
\Sigma = \begin{bmatrix} \ 1& 0.5\ \\ \ 0.5& 1\ \end{bmatrix}; \ 
\Sigma = \begin{bmatrix} \ 1& 0.8\ \\ \ 0.8& 1\ \end{bmatrix}.
$$

最左边的图显示的是熟悉的标准正态分布，并且可以看到，随着 $\Sigma$ 中非对角线元素的增加，密度函数变得更加“压缩”到 $45^\circ$ 线 (由 $x_1 = x_2$ 给出)。当观察这三个密度函数的等高线时，可以更清楚地看到这一点：

![[gaussian_contour1.svg]]

下面是另外一组由不同的 $\Sigma$ 生成的例子：

![[gaussian_contour2.svg]]

上面的图分别使用了

$$
\Sigma = \begin{bmatrix} \ 1& -0.5\ \\ \ -0.5& 1\ \end{bmatrix}; \ 
\Sigma = \begin{bmatrix} \ 1& -0.8\ \\ \ -0.8& 1\ \end{bmatrix}; \ 
\Sigma = \begin{bmatrix} \ 3& 0.8\ \\ \ 0.8& 1\ \end{bmatrix}.
$$

从最左边和中间的图可以看到，通过减小协方差矩阵的非对角线元素，密度函数再次变得“压缩”，但方向相反。最后值得一提的是，当改变参数时，等高线通常会形成椭圆（最右边的图显示了一个例子）。

作为最后一组例子，通过固定 $\Sigma = I$，并改变 $\mu$，也可以移动密度函数的均值。

![[gaussian_shift.svg]]

上面的图是使用 $\Sigma = I$ 生成的，并且 $\mu$ 分别为

$$
\mu = \begin{bmatrix} 
		\ 1 \ \\
		\ 0 \ 
	\end{bmatrix}; \ 
\mu = \begin{bmatrix} 
		\ -0.5 \ \ \\
		\ 0 
	\end{bmatrix}; \ 
\mu = \begin{bmatrix} 
		-1 \\
		\ -1.5 \ \ 
	\end{bmatrix}.
$$

### 4.1.2 高斯判别分析模型

lorem

### 4.1.3 讨论：GDA 与逻辑回归

lorem

## 4.2 朴素贝叶斯 (选读)

lorem

### 4.2.1 拉普拉斯平滑

lorem

### 4.2.2 文本分类的事件模型

lorem

| [[chapter3_generalized_linear_model\|上一章]] | [[CS229_CN/index\|目录]] | 下一章 |
| :----------------------------------------: | :--------------------: | :-: |
