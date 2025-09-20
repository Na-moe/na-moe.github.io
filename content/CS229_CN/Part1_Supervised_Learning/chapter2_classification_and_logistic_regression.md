---
title: 第二章 分类与逻辑回归
---
现在我们转向分类问题。这与回归问题类似，主要区别在于需要预测的目标变量 $y$ 只能取有限的离散值。目前，我们将重点讨论 **二元分类 (binary classification)** 问题，其中 $y$ 的取值仅限于 $0$ 和 $1$.（这里讨论的大部分内容也适用于多类别分类情况。）例如，在构建垃圾邮件分类器时， 可以代表一封电子邮件的某些特征，而 $y$ 则表示该邮件是否为垃圾邮件（垃圾邮件为 $1$，非垃圾邮件为 $0$）。通常，$0$ 被称为 **负类 (negetive class)**，$1$ 被称为 **正类 (positive class)**，有时也用符号 “$-$” 和 “$+$” 表示。对于给定的输入特征 $x^{(i)}$，其对应的 $y^{(i)}$ 被称为该训练样本的 **标签 (label)**。

## 2.1 逻辑回归

在处理分类问题时，我们可以暂时忽略目标变量 $y$ 是离散值这一特性，并沿用之前的线性回归算法来尝试预测给定输入 $x$ 的 $y$ 值。然而，很容易构造出这种方法表现极差的例子。直觉上，由于 $y$ 只能取 $\{0, 1\}$ 中的值，模型的输出 $h_\theta(x)$ 取大于 1 或小于 0 的值是没有意义的。为了解决这一问题，我们需要改变假设函数 $h_\theta(x)$ 的形式。具体而言，我们将选择：

$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1+e^{-\theta^T x}},
$$

其中

$$
g(z) = \frac{1}{1+e^{-z}},
$$

称为 **逻辑函数 (logistic function)** 或 **S 形函数 (sigmoid function)**。下面是 $g(z)$ 的图像：

![[CS229_CN/Part1_Supervised_Learning/figs/sigmoid.svg|500]]

注意到 $g(z)$ 在 $z \to \infty$ 时趋于 1，在 $z \to -\infty$ 时趋于 0。因此 $h(x)$ 始终介于 0 和 1 之间。和之前一样，这里约定 $x_0 = 1$, 从而有 $\theta^T x = \theta_0 + \sum_{j=1}^d \theta_j x_j$.

现在先将 $g$ 的形式视为一个已知条件。其他从 0 平滑增加到 1 的函数也可以使用，但出于一些原因 (稍后讨论广义线性模型 (GLMs) 和生成学习算法时会看到)，选择 sigmoid 函数是相当自然的。在进一步展开之前，记 sigmoid 函数导数为 $g'$, 它有一个有用的性质：

$$
\begin{aligned}
	g'(z) 
		&= \frac{d}{dz} \frac{1}{1+e^{-z}} \\
		&= \frac{1}{(1+e^{-z})^2} (e^{-z}) \\
		&= \frac{1}{1+e^{-z}} \cdot \left(1 - \frac{1}{1+e^{-z}}\right) \\
		&= g(z)(1 - g(z)).
\end{aligned}
$$

那么，给定这样的逻辑回归模型，我们如何为其拟合 $\theta$ 呢？参照我们之前所见，最小二乘回归在一定假设下可以作为最大似然估计的一种形式。因此，我们也将为分类模型设定一组概率假设，然后通过最大似然估计的方式来拟合参数。

假设

$$
\begin{aligned}
	P(y=1 \mid x; \theta) &= h_\theta(x) \\
	P(y=0 \mid x; \theta) &= 1 - h_\theta(x)
\end{aligned}
$$

注意到上述两个概率表达式可以合并为一个更紧凑的形式

$$
p(y \mid x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}
$$

假设 $n$ 个训练样本是独立生成的，那么参数的似然可以写成

$$
\begin{aligned}
	L(\theta) 
		&= p(\vec{y} \mid X; \theta) \\
		&= \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta) \\
		&= \prod_{i=1}^n (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}}
\end{aligned}
$$

和之前一样，最大化对数似然会更容易推导：

$$
\begin{equation}
	\ell(\theta) = \log L(\theta) = \sum_{i=1}^n y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) \tag{2.1}
\end{equation}
$$

^eqc2eq1

那么，如何最大化这个似然函数呢呢？类似于线性回归的推导过程，我们可以采用梯度上升法。以向量形式表示，参数的更新规则为 $\theta := \theta + \alpha \nabla_\theta \ell(\theta)$. (注意更新公式中的正号，因为现在是在最大化函数，而不是最小化函数。) 接下来，我们将从一个训练样本 $(x, y)$ 出发，推导随机梯度上升规则的导数：

$$
\begin{align}
	\frac{\partial}{\partial \theta_j} \ell(\theta) 
		&= \left(y \frac{1}{g(\theta^T x)} - (1-y) \frac{1}{1-g(\theta^T x)}\right) \frac{\partial}{\partial \theta_j} g(\theta^T x) \\
	    &= \left(y \frac{1}{g(\theta^T x)} - (1-y) \frac{1}{1-g(\theta^T x)}\right) g(\theta^T x)(1-g(\theta^T x)) \frac{\partial}{\partial \theta_j} \theta^T x \\
	    &= (y(1-g(\theta^T x)) - (1-y)g(\theta^T x)) x_j \\
	    &= (y - g(\theta^T x)) x_j \tag{2.1}
\end{align}
$$

^eqc2eq2

上面的推导利用了 $g'(z) = g(z)(1-g(z))$ 这一点。这给出了随机梯度上升规则：

$$
\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}
$$

如果将推导出的逻辑回归更新规则与最小均方更新规则进行比较，会发现它们在形式上是相同的；但这并不是同一个算法，因为这里的 $h_\theta(x^{(i)})$ 是 $\theta^T x^{(i)}$ 的非线性函数。尽管如此，对于一个截然不同的算法和学习问题，却得到了相同的更新规则，这确实有些令人惊讶。这仅仅是巧合吗？抑或是背后存在更深层的原因？我们将在讨论广义线性模型 (GLM) 时解答这个问题。

**备注 2.1.1.** 同一个损失函数的另一种表示方式也很有用，特别是在第 7.1 节研究非线性模型时。

设逻辑损失函数 $\ell_{\text{logistic}}: \mathbb{R} \times \{0, 1\} \to \mathbb{R}_{\ge 0}$ 定义为

$$
\ell_{\text{logistic}}(t, y) \triangleq y \log(1 + \exp(-t)) + (1 - y) \log(1 + \exp(t)). \tag{2.3}
$$

通过代入 $h_\theta(x) = 1/(1 + e^{-\theta^T x})$，可以验证负对数似然 (方程 [[chapter2_classification_and_logistic_regression#^eqc2eq1|(2.1)]] 中 $\ell(\theta)$ 的负值) 可以改写为

$$
-\ell(\theta) = \ell_{\text{logistic}}(\theta^T x, y).
$$

通常 $\theta^T x$ 或 $t$ 称为 *logit*. 稍作运算可得

$$
\begin{align}
	\frac{\partial \ell_{\text{logistic}}(t, y)}{\partial t} 
		&= y \frac{-\exp(-t)}{1 + \exp(-t)} + (1 - y) \frac{1}{1 + \exp(-t)} \tag{2.5} \\
		&= \frac{1}{1 + \exp(-t)} - y. \tag{2.6}
\end{align}
$$

然后，使用链式法则，得到

$$
\begin{align}
	\frac{\partial}{\partial \theta_j} \ell(\theta) 
		&= - \frac{\partial \ell_{\text{logistic}}(t, y)}{\partial t} \cdot \frac{\partial t}{\partial \theta_j} \tag{2.7} \\
		&= (y-1/(1+\exp(-t))) \cdot x_j = (y - h_\theta(x)) x_j, \tag{2.8}
\end{align}
$$

这与方程 [[chapter2_classification_and_logistic_regression#^eqc2eq2|(2.2)]] 的推导是一致的。在第 7.1 节中，会看到这种观点可以扩展到非线性模型。
## 2.2 离题：感知机学习算法

## 2.3 多类别分类

## 2.4 最大化 $\ell(\theta)$ 的另一种算法
