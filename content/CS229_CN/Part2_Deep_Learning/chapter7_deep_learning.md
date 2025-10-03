---
title: 第 7 章 深度学习
---
| [[CS229_CN/Part2_Deep_Learning/index\|上一章]] | [[CS229_CN/index#目录\|目录]] | [[chapter7_deep_learning\|下一章]] |
| :-----------------------------------------: | :-----------------------: | :-----------------------------: |

## 7.1 使用非线性模型的监督学习

在监督学习设置中 (从输入 $x$ 预测 $y$ )，假设我们的模型/假设是 $h_\theta(x)$. 过去的章节里考虑了 $h_\theta(x) = \theta^T x$ (线性回归) 或 $h_\theta(x) = \theta^T \phi(x)$ (其中 $\phi(x)$ 是特征映射) 的情况。这两个模型的共同点是它们对于参数 $\theta$ 是线性的。接下来，我们将考虑学习对于参数 $\theta$ 和输入 $x$ 都是 **非线性 (non-linear)** 的一般模型族。最常见的非线性模型是神经网络，我们将在下一节开始定义。对于本节，将 $h_\theta(x)$ 视为抽象的非线性模型即可。[^1]

假设 $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$ 是训练样本。我们将定义非线性模型及其学习的损失/成本函数。

### 回归问题

为简单起见，我们从输出是实数的情况开始，即 $y^{(i)} \in \mathbb{R}$, 因此模型 $h_\theta(x)$ 也输出实数 $h_\theta(x) \in \mathbb{R}$. 将学习第 $i$ 个样本 $(x^{(i)}, y^{(i)})$ 的最小二乘成本函数定义为

$$
J^{(i)}(\theta) = \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2, \tag{7.1}
$$

数据集的均方成本函数定义为

^eq7-2
$$
J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta), \tag{7.2}
$$

这与线性回归中的定义相同，为了与约定一致，我们在成本函数前引入了常数 $1/n$. 注意，将成本函数乘以一个标量不会改变成本函数的局部最小点或全局最小点。另请注意，尽管成本函数的形式也是均方损失，但 $h_\theta(x)$ 的参数化与线性回归的情况不同。在后文中，“损失”和“成本”这两个词会互换使用。

### 二元分类

接下来定义二元分类的模型和损失函数。假设输入 $x \in \mathbb{R}^d$. 令 $\bar{h}_\theta: \mathbb{R}^d \to \mathbb{R}$ 是一个参数化模型 (逻辑线性回归中 $\theta^T x$ 的类比)。将输出 $\bar{h}_\theta(x) \in \mathbb{R}$ 称为 logit. 类似于第 [[chapter2_classification_and_logistic_regression#2.1 逻辑回归|2.1]] 节，使用 logistic 函数 $g(\cdot)$ 将 logit $\bar{h}_\theta(x)$ 转换为概率 $h_\theta(x) \in [0, 1]$:

$$
h_\theta(x) = g(\bar{h}_\theta(x)) = 1 / (1 + \exp(-\bar{h}_\theta(x))). \tag{7.3}
$$

使用以下方式建模给定 $x$ 和 $\theta$ 的 $y$ 的条件分布：

$$
\begin{aligned}
    P(y = 1 \mid x; \theta) &= h_\theta(x) \\
    P(y = 0 \mid x; \theta) &= 1 - h_\theta(x)
\end{aligned}
$$

按照第 [[chapter2_classification_and_logistic_regression#2.1 逻辑回归|2.1]] 节中的相同推导并使用备注 [[chapter2_classification_and_logistic_regression#^rmk2-1-1|2.1.1]] 中的推导，负对数似然损失函数等于：

$$
J^{(i)}(\theta) = -\log p(y^{(i)} \mid x^{(i)}; \theta) = \ell_{\text{logistic}}(\bar{h}_\theta(x^{(i)}), y^{(i)}) \tag{7.4}
$$

如公式 [[chapter7_deep_learning#^eq7-2|(7.2)]] 中所示，总损失函数也定义为对单个训练样本的损失函数的平均值，$J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta)$.

### 多类别分类

按照第 [[chapter2_classification_and_logistic_regression#2.3 多类别分类|2.3]] 节，考虑响应变量 $y$ 可以取 $k$ 个值之一的分类问题，即 $y \in \{1, 2, \dots, k\}$. 令 $\bar{h}_\theta: \mathbb{R}^d \to \mathbb{R}^k$ 是一个参数化模型。将输出 $\bar{h}_\theta(x) \in \mathbb{R}^k$ 称为 logits. 每个 logit 对应于 $k$ 个类别之一的预测。类似于第 [[chapter2_classification_and_logistic_regression#2.3 多类别分类|2.3]] 节，使用 $\text{softmax}$ 函数将 logits $\bar{h}_\theta(x)$ 转换为一个非负且和为 $1$ 的概率向量：

$$
P(y = j \mid x; \theta) = \frac{\exp(\bar{h}_\theta(x)_j)}{\sum_{s=1}^k \exp(\bar{h}_\theta(x)_s)}, \tag{7.5}
$$

其中 $\bar{h}_\theta(x)_s$ 表示 $\bar{h}_\theta(x)$ 的第 $s$ 个坐标。

类似于第 [[chapter2_classification_and_logistic_regression#2.3 多类别分类|2.3]] 节，单个训练样本 $(x^{(i)}, y^{(i)})$ 的损失函数是其负对数似然：

$$
J^{(i)}(\theta) = -\log p(y^{(i)} \mid x^{(i)}; \theta) = -\log \left( \frac{\exp(\bar{h}_\theta(x^{(i)})_{y^{(i)}})}{\sum_{s=1}^k \exp(\bar{h}_\theta(x^{(i)})_s)} \right). \tag{7.6}
$$

使用第 [[chapter2_classification_and_logistic_regression#2.3 多类别分类|2.3]] 节的符号，可以简单地抽象地写成：

$$
J^{(i)}(\theta) = \ell_{\text{ce}}(\bar{h}_\theta(x^{(i)}), y^{(i)}).
$$

损失函数也定义为单个训练样本的损失函数的平均值，$J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta)$.

还注意到，上述方法也可以推广到任何条件概率模型，其中有一个关于 $y$ 的指数族分布 $\text{Exponential-Family}(y; \eta)$, 其中 $\eta = \bar{h}_\theta(x)$ 是一个参数化的非线性函数 $x$. 然而，最广泛的情况还是上面讨论的三种情况。

### 优化器 (SGD)

通常使用梯度下降 (GD)、随机梯度下降 (SGD) 或其变体来优化损失函数 $J(\theta)$. GD 的更新规则可以写为[^2]

$$
\theta := \theta - \alpha \nabla_\theta J(\theta) \tag{7.8}
$$

其中 $\alpha > 0$ 通常称为学习率或步长。 接下来，介绍一个新的 SGD (算法[[chapter7_deep_learning#^algo1|1]])，它与前面所讲的略有不同。

---

**算法 1** 随机梯度下降 ^algo1

---

1: 超参数: 学习率 $\alpha$, 总迭代次数 $n_\text{iter}$.

2: 随机初始化 $\theta$.

3: **for** $i=1$ 到 $n_\text{iter}$

4: $\qquad$从 $\{1, ..., n\}$ 中均匀采样 $j$, 使用下式更新 $\theta$

$$
\theta := \theta - \alpha \nabla_\theta J^{(j)}(\theta) \tag{7.9}
$$

---

通常，因为硬件并行化，同时计算 $B$ 个样本关于参数 $\theta$ 的梯度比单独计算 $B$ 个梯度要快。 因此，深度学习中更常用的是小批量随机梯度下降，如算法 [[chapter7_deep_learning#^algo2|2]] 所示。还有一些 SGD 或小批量 SGD 的变体，它们使用略微不同的采样方案。

---

**算法 2** 小批量随机梯度下降 ^algo2

---

1: 超参数: 学习率 $\alpha$, 批量大小 $B$, 迭代次数 $n_\text{iter}$.

2: 随机初始化 $\theta$.

3: **for** $i=1$ 到 $n_\text{iter}$

4: $\qquad$从 $\{1, ..., n\}$ 中不放回地均匀采样 $B$ 个样本 $j_1, \dots, j_B$, 使用下式更新 $\theta$

$$
\theta := \theta - \frac{\alpha}{B} \sum_{k=1}^B \nabla_\theta J^{(j_k)}(\theta) \tag{7.10}
$$

---

使用这些通用算法，典型的深度学习模型通过以下步骤进行学习。 1. 定义神经网络参数化 $h_\theta(x)$，将在第 [[chapter7_deep_learning#7.2 神经网络|7.2]] 节中介绍，2. 编写反向传播算法以有效计算损失函数 $J^{(j)}(\theta)$ 的梯度，这将在第 [[chapter7_deep_learning#7.4 反向传播|7.4]] 节中介绍，以及 3. 使用损失函数 $J(\theta)$ 运行 SGD 或小批量 SGD (或其他基于梯度的优化器)。

## 7.2 神经网络

lorem

## 7.3 现代神经网络的模块

lorem

## 7.4 反向传播

lorem

| [[CS229_CN/Part2_Deep_Learning/index\|上一章]] | [[CS229_CN/index#目录\|目录]] | [[chapter7_deep_learning\|下一章]] |
| :-----------------------------------------: | :-----------------------: | :-----------------------------: |

[^1]: 如果需要一个具体的例子，可以考虑模型 $h_\theta(x) = \theta_1^2 x_1^2 + \theta_2^2 x_2^2 + \dots + \theta_d^2 x_d^2$, 尽管它不是神经网络。

[^2]: 回顾一下，如之前所定义，使用符号 “$a := b$” 表示一个操作 (在计算机程序中)，其中将变量 $a$ 的值设置为 $b$. 换句话说，此操作用 $b$ 的值覆盖 $a$. 相比之下，当断言事实陈述 $a$ 的值等于 $b$ 的值时，将写成 “$a = b$”.
