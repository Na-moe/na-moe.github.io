---
title: 第 3 章 广义线性模型
---

| [[chapter2_classification_and_logistic_regression\|上一章]] | [[CS229_CN/index\|目录]] | [[chapter4_generative_learning_algorithms\|下一章]] |
| :------------------------------------------------------: | :--------------------: | :----------------------------------------------: |

到目前我们已经讨论了一个回归的例子和一个分类的例子。在回归示例中，有 $y|x; \theta \sim \mathcal{N}(\mu, \sigma^2)$, 在分类示例中，有 $y|x; \theta \sim \text{Bernoulli}(\phi)$, 其中 $\mu$ 和 $\phi$ 是 $x$ 和 $\theta$ 的函数。本章将揭示这两种方法都是更广泛的模型族——称为广义线性模型 (GLMs) ——的特例。[^1] 我们还将展示广义线性模型族中的其他模型如何推导并应用于其他分类和回归问题。

## 3.1 指数族

为了逐步了解广义线性模型，首先定义指数族分布。如果一类分布可以写成以下形式，则称其为指数族：

^eq3-1
$$
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta)) \tag{3.1}
$$

这里，$\eta$ 称为 **自然参数 (natural parameter)** (也称为 **典范参数 (canonical parameter)**)；$T(y)$ 是 **充分统计量 (sufficient statistic)** (对于所考虑的分布，通常有 $T(y)=y$)；而 $a(\eta)$ 是 **对数配分函数 (log partition function)**。量 $e^{-a(\eta)}$ 实际上起着归一化常数的作用，确保分布 $p(y; \eta)$ 在 $y$ 上的和或积分等于 $1$.

固定的 $T$, $a$ 和 $b$ 定义了一个由 $\eta$ 参数化的*族 (family)* (或分布集)；随着 $\eta$ 的变化，将得到该族中的不同分布。

现在展示伯努利分布和高斯分布是指数族分布的示例。具有均值 $\phi$ 的伯努利分布，记为 $\text{Bernoulli}(\phi)$, 其给出了一个在 $y \in \{0, 1\}$ 上的分布，满足 $p(y=1; \phi) = \phi$; $p(y=0; \phi) = 1-\phi$. 随着 $\phi$ 的变化，可以得到具有不同均值的伯努利分布。我们接下来证明，改变 $\phi$ 所得到的这些伯努利分布均属于指数族；也就是说，存在一种 $T, a, b$ 的选择，使得公式 [[chapter3_generalized_linear_model#^eq3-1|(3.1)]] 恰好成为伯努利分布。

将伯努利分布写为：

$$
\begin{aligned}
    p(y; \phi) 
	    &= \phi^y (1-\phi)^{1-y} \\
	    &= \exp\left(y \log \phi + (1-y) \log(1-\phi)\right) \\
	    &= \exp\left(\left(\log\left(\frac{\phi}{1-\phi}\right)\right)y + \log(1-\phi)\right).
\end{aligned}
$$

因此，自然参数由 $\eta = \log(\phi/(1-\phi))$ 给出。有趣的是，如果通过求解 $\phi$ 关于 $\eta$ 的表达式来反转这个定义，得到 $\phi = 1/(1+e^{-\eta})$. 这是熟悉的 $\text{sigmoid}$ 函数！在将逻辑回归推导为广义线性模型时，这将再次出现。为了完成伯努利分布作为指数族分布的形式化，还需要以下各项：

$$
\begin{aligned}
    T(y) &= y \\
    a(\eta) &= -\log(1-\phi) \\
	    &= \log(1+e^\eta) \\
    b(y) &= 1
\end{aligned}
$$

这表明伯努利分布可以通过选择适当的 $T, a, b$ 从而写成公式 [[chapter3_generalized_linear_model#^eq3-1|(3.1)]] 的形式。

接下来考虑高斯分布。回想一下，在线性回归推导中，$\sigma^2$ 的值对最终选择的 $\theta$ 和 $h_\theta(x)$ 没有影响。因此，可以在不改变任何内容的情况下选择任意的 $\sigma^2$ 值。为了简化下面的推导，令 $\sigma^2 = 1$.[^2] 有：

$$
\begin{aligned}
    p(y; \mu) 
	    &= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}(y-\mu)^2\right) \\
	    &= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}y^2 + \mu y - \frac{1}{2}\mu^2\right) \\
	    &= \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}y^2\right) \cdot \exp\left(\mu y - \frac{1}{2}\mu^2\right).
\end{aligned}
$$

因此， 高斯分布属于指数族，其中

$$
\begin{aligned}
    \eta &= \mu \\
    T(y) &= y \\
    a(\eta) &= \mu^2/2 \\
	    &= \eta^2/2 \\
    b(y) &= (1/\sqrt{2\pi})\exp(-y^2/2).
\end{aligned}
$$

还有许多其他分布也属于指数族：多项分布 (稍后将看到)、泊松分布 (用于建模计数数据；另请参阅问题集)、伽马分布和指数分布 (用于建模连续非负随机变量，例如时间间隔)、Beta 分布和 Dirichlet 分布 (用于概率分布) 等等。下一节中将描述，当 $y$ (给定 $x$ 和 $\theta$) 来自这些分布时，构建模型的一般“配方”。

## 3.2 构造广义线性模型

假设希望构建一个模型来估计在给定小时内到达商店的顾客数量 (或网站的页面浏览量 $y$), 基于某些特征 $x$, 例如商店促销、近期广告、天气、星期几等等。已知泊松分布通常能很好地模拟访客数量。知道了这一点，如何为这个问题构建模型？幸运的是，泊松分布是指数族分布，因此可以应用广义线性模型。在本节中，将描述一种构建用于解决此类问题的广义线性模型的方法。

更一般地，考虑一个分类或回归问题，希望预测某个随机变量 $y$ 作为 $x$ 的函数的值。为了推导针对该问题的广义线性模型，将对给定 $x$ 的 $y$ 的条件分布和模型做出以下三个假设：

1. $y|x; \theta \sim \text{ExponentialFamily}(\eta)$. 也就是说，给定 $x$ 和 $\theta$, $y$ 的分布遵循参数为 $\eta$ 的某个指数族分布；
2. 给定 $x$, 目标是预测 $T(y)$ 的期望值。在大多数示例中，$T(y)=y$, 因此这意味着希望学习到的假设 $h$ 的输出预测 $h(x)$ 满足 $h(x) = E[y|x]$. (注意，这个假设在逻辑回归和线性回归中对于 $h_\theta(x)$ 的选择是满足的。例如，在逻辑回归中，$h_\theta(x) = p(y=1|x; \theta) = 0 \cdot p(y=0|x; \theta) + 1 \cdot p(y=1|x; \theta) = E[y|x; \theta]$.)；
3. 自然参数 $\eta$ 和输入 $x$ 线性相关：$\eta = \theta^T x$. (或者，如果 $\eta$ 是向量值，则 $\eta_i = \theta_i^T x$.)

第三个假设可能看起来最不合理，最好将其视为设计广义线性模型的“设计选择”，而不是一个固有的假设。这三个假设/设计选择将允许推导出非常优雅的一类学习算法，即广义线性模型，它们具有许多理想的特性，例如易于学习。此外，由此产生的模型对于建模不同类型的 $y$ 分布非常有效；例如，很快将展示逻辑回归和普通最小二乘都可以作为广义线性模型推导出来。

### 3.2.1 普通最小二乘

为了展示普通最小二乘是广义线性模型族的一个特例，考虑目标变量 $y$ (在广义线性模型术语中也称为 **响应变量 (response variable)**)是连续的情况，并将给定 $x$ 的 $y$ 的条件分布建模为高斯分布 $N(\mu, \sigma^2)$. (这里 $\mu$ 可能取决于 $x$.) 因此，令上面的 $\text{ExponentialFamily}(\eta)$ 分布为 高斯分布。如前所述，在将高斯分布形式化为指数族分布时，有 $\mu = \eta$. 因此，有

$$
\begin{aligned}
    h_\theta(x) 
	    &= E[y|x; \theta] \\
	    &= \mu \\
	    &= \eta \\
	    &= \theta^T x.
\end{aligned}
$$

第一个等号来自于上面的假设 $2$；第二个等号来自于 $y|x; \theta \sim N(\mu, \sigma^2)$, 因此其期望值由 $\mu$ 给出；第三个等号来自于假设 $1$ (以及之前推导中表明在将高斯分布公式化为指数族分布时 $\mu = \eta$); 最后一个等号来自于假设 $3$.

### 3.2.2 逻辑回归

现在考虑逻辑回归。对于二分类问题，有 $y \in \{0, 1\}$. 鉴于 $y$ 是二值变量，自然而然地选择伯努利分布族来建模给定 $x$ 的 $y$ 的条件分布。在将伯努利分布形式化为指数族分布时，有 $\phi = 1/(1 + e^{-\eta})$. 此外，注意如果 $y|x; \theta \sim \text{Bernoulli}(\phi)$, 则 $E[y|x; \theta] = \phi$. 因此，按照与普通最小二乘类似的推导，得到：

$$
\begin{aligned}
    h_\theta(x) 
	    &= E[y|x; \theta] \\
	    &= \phi \\
	    &= 1/(1 + e^{-\eta}) \\
	    &= 1/(1 + e^{-\theta^T x})
\end{aligned}
$$

因此，这给出了形式为 $h_\theta(x) = 1/(1 + e^{-\theta^T x})$ 的假设函数。如果之前想知道如何得到逻辑函数的形式 $1/(1 + e^{-z})$，这就是一个答案：一旦假设给定 $x$ 的 $y$ 服从伯努利分布，它就作为广义线性模型和指数族分布定义的必然结果出现了。

这里引入一些额外的术语，将自然参数映射到分布均值的函数 $g$ ($g(\eta) = E[T(y); \eta]$), 称为**典范响应函数 (canonical response function)**。其逆函数 $g^{-1}$ 称为 **典范连接函数 (canonical link function)**。因此，高斯族分布的典范响应函数就是恒等函数；伯努利分布的典范响应函数是逻辑函数。[^3]

| [[chapter2_classification_and_logistic_regression\|上一章]] | [[CS229_CN/index\|目录]] | [[chapter4_generative_learning_algorithms\|下一章]] |
| :------------------------------------------------------: | :--------------------: | :----------------------------------------------: |

[^1]: 本章材料受到 Michael I. Jordan 的 *Learning in graphical models* (未出版的书稿) 以及 McCullagh 和 Nelder 的 *Generalized Linear Models (2nd ed.)* 的启发。

[^2]: 如果将 $\sigma^2$ 作为一个变量，高斯分布也仍可以推导为指数族，其中 $\eta \in \mathbb{R}^2$ 现在是一个二维向量，它取决于 $\mu$ 和 $\sigma$. 然而，出于广义线性模型的目的，$\sigma^2$ 参数也可以通过考虑指数族的一个更一般的定义来处理：$p(y; \eta, \tau) = b(a, \tau) \exp((\eta^T T(y) - a(\eta))/c(\tau))$. 这里，$\tau$ 称为 **散布参数 (dispersion parameter)**，对于 高斯分布，$c(\tau) = \sigma^2$; 但考虑到上面的简化，这里不需要更一般的定义。

[^3]: 许多文献使用 $g$ 表示连接函数，$g^{-1}$ 表示响应函数；但这里使用的符号继承自早期的机器学习文献，将与课程其余部分使用的符号更一致。
