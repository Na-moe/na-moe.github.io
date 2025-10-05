---
title: "习题集 #1: 监督学习"
---
此为 [2023-夏](https://github.com/JustinaZ/CS229/blob/main/CS229_2023Summer_ProblemSet1.pdf) 版的习题集#1

注意：解答中不能使用机器学习专用库 (如 `scikit-learn`)

---

### 1. \[40 分\] 线性分类器 (逻辑回归与高斯判别分析)

在本问题中，我们将讨论至今课程中涵盖的两种概率性线性分类器。第一种是判别式线性分类器：逻辑回归。第二种是生成式线性分类器：高斯判别分析 (GDA)。这两种算法都寻找一个将数据划分为两类的线性决策边界，但基于不同的假设。本题的目标是让您更深入地理解这两种算法的异同及其优缺点。

针对此问题，我们将使用以下文件提供的两个数据集：

i. `data/ds1_{train,valid}.csv`
ii. `data/ds2_{train,valid}.csv`

每个文件包含 m 个样本，每行一个样本 $(x^{(i)}, y^{(i)})$。具体来说，第 i 行包含 $x_1^{(i)} \in \mathbb{R}$、$x_2^{(i)} \in \mathbb{R}$ 和 $y^{(i)} \in \{0,1\}$ 这几列。在接下来的子问题中，我们将研究使用逻辑回归和高斯判别分析对这两个数据集进行二元分类。

(a) \[10分\] 在课堂上我们看到了逻辑回归的平均经验损失：

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right],
$$

其中 $y^{(i)} \in \{0,1\}$, $h_\theta(x) = g(\theta^T x)$, 且 $g(z) = \frac{1}{1 + e^{-z}}$.

求此函数的 Hessian 矩阵 $H$, 并证明对于任意向量 $z$, 都有 $z^T H z \geq 0$ 成立。

**提示：** 您可以从证明 $\sum_i \sum_j z_i x_i x_j z_j = (x^T z)^2 \geq 0$ 开始。同时回顾 $g'(z) = g(z)(1 - g(z))$.

**注：** 这是证明矩阵 $H$ 是半正定矩阵 (记为 $H \succeq 0$) 的标准方法之一。这意味着 $J$ 是凸函数，并且除了全局最小值外没有其他局部最小值。如果您有其他证明 $H \succeq 0$ 的方法，也欢迎使用。

(b) \[5分\] **编程题:** 请按照 `src/p01b_logreg.py` 中的说明，使用牛顿法训练一个逻辑回归分类器。从 $\theta = \mathbf{0}$ 开始，运行牛顿法直到 $\theta$ 的更新量足够小：具体来说，训练直到第一次满足 $\|\theta_k - \theta_{k-1}\|_1 < \varepsilon$ 的迭代 $k$, 其中 $\varepsilon = 1 \times 10^{-5}$. 请确保将您模型的预测结果提交到代码中指定的文件。

(c) \[5分\] 回顾在高斯判别分析中，我们通过以下方程对 $(x, y)$ 的联合分布进行建模：

$$
\begin{aligned}
p(y) &=
\begin{cases}
\phi & \text{若 } y = 1 \\
1 - \phi & \text{若 } y = 0
\end{cases} \\
p(x|y = 0) &= \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0)\right) \\
p(x|y = 1) &= \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1)\right),
\end{aligned}
$$

---

### 2. \[30 分\] 正定矩阵

若矩阵 $A \in \mathbb{R}^{n \times n}$ 满足 $A = A^T$ 且对所有 $x \in \mathbb{R}^n$ 有 $x^T A x \geq 0$, 则称其为*半正定*矩阵，记作 $A \succeq 0$. 若矩阵 $A$ 满足 $A = A^T$ 且对所有非零向量 $x \neq 0$ 有 $x^T A x > 0$, 则称其为*正定*矩阵，记作 $A \succ 0$. 正定矩阵的最简范例是单位矩阵 $I$ (对角线元素为 $1$、非对角线元素为 $0$ 的对角矩阵)，其满足 $x^T I x = \|x\|_2^2 = \sum_{i=1}^n x_i^2$.

(a) 设 $z \in \mathbb{R}^n$ 为一 $n$ 维向量。证明 $A = z z^T$ 是半正定矩阵。

(b) 设 $z \in \mathbb{R}^n$ 为一*非零* $n$ 维向量，且 $A = z z^T$. 问：$A$ 的零空间是什么？$A$ 的秩是多少？

(c) 设 $A \in \mathbb{R}^{n \times n}$ 为半正定矩阵，$B \in \mathbb{R}^{m \times n}$ 为任意矩阵，其中 $m, n \in \mathbb{N}$. 问：$B A B^T$ 是否为半正定矩阵？若是，请予以证明；若否，请给出具体的 $A$ 与 $B$ 作为反例。

---

### 3. \[25 分\] 特征向量、特征值与谱定理

$n \times n$ 矩阵 $A \in \mathbb{R}^{n \times n}$ 的特征值是其特征多项式 $p_A(\lambda) = \det(\lambda I - A)$ 的根 (通常可能为复数)。特征值也可定义为满足 $A x = \lambda x$ (其中 $x \in \mathbb{C}^n$) 的标量 $\lambda \in \mathbb{C}$. 我们称这样的二元组 $(x, \lambda)$ 为特征向量-特征值对。在本问题中，我们使用符号 $\operatorname{diag}(\lambda_1, \dots, \lambda_n)$ 表示对角元素为 $\lambda_1, \dots, \lambda_n$ 的对角矩阵，即

$$
\operatorname{diag}(\lambda_1, \dots, \lambda_n) = 
\begin{bmatrix}
\lambda_1 & 0 & 0 & \cdots & 0 \\
0 & \lambda_2 & 0 & \cdots & 0 \\
0 & 0 & \lambda_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \lambda_n
\end{bmatrix}.
$$

(a) 假设矩阵 $A \in \mathbb{R}^{n \times n}$ 可对角化，即存在可逆矩阵 $T \in \mathbb{R}^{n \times n}$ 和对角矩阵 $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$, 使得 $A = T \Lambda T^{-1}$. 记 $t^{(i)}$ 为 $T$ 的列向量，即 $T = [t^{(1)} \cdots t^{(n)}]$, 其中 $t^{(i)} \in \mathbb{R}^n$. 证明 $A t^{(i)} = \lambda_i t^{(i)}$, 从而说明 $A$ 的*特征值/特征向量*对为 $(t^{(i)}, \lambda_i)$.

若矩阵 $U \in \mathbb{R}^{n \times n}$ 满足 $U^T U = I$, 则称其为正交矩阵。**谱定理** (可能是线性代数中最重要的定理之一) 指出：若 $A \in \mathbb{R}^{n \times n}$ 是对称矩阵 (即 $A = A^T$ ), 则 $A$ 可*通过实正交矩阵对角化*。即，存在对角矩阵 $\Lambda \in \mathbb{R}^{n \times n}$ 和正交矩阵 $U \in \mathbb{R}^{n \times n}$, 使得 $U^T A U = \Lambda$, 或等价地，

$$
A = U \Lambda U^T.
$$

记 $\lambda_i = \lambda_i(A)$ 为 $A$ 的第 $i$ 个特征值。

(b) 设 $A$ 为对称矩阵。证明：若 $U = [u^{(1)} \cdots u^{(n)}]$ 为正交矩阵 (其中 $u^{(i)} \in \mathbb{R}^n$), 且 $A = U \Lambda U^T$, 则 $u^{(i)}$ 是 $A$ 的特征向量，且满足 $A u^{(i)} = \lambda_i u^{(i)}$, 其中 $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$.

(c) 证明：若 $A$ 是半正定矩阵，则对每个 $i$, 有 $\lambda_i(A) \geq 0$.