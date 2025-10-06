---
title: "习题集 #1: 监督学习"
---
此为 [2020-夏](https://github.com/JustinaZ/CS229/blob/main/CS229_2023Summer_ProblemSet1.pdf) 版的习题集#1

注意：解答中不能使用机器学习专用库 (如 `scikit-learn`)

所涉及的数据及初始代码：[下载]()

---

### 1. \[40 分\] 线性分类器 (逻辑回归与高斯判别分析)

本题涵盖了目前课程中涉及的两种概率性线性分类器。第一种是判别式线性分类器：逻辑回归。第二种是生成式线性分类器：高斯判别分析 (GDA)。这两种算法均通过寻找线性决策边界将数据划分为两个类别，但基于不同的假设。本题旨在帮助深入理解这两种算法的异同及其优缺点。

本题将使用以下文件中的两个数据集及初始代码：

- `src/linearclass/ds1_{train,valid}.csv`
- `src/linearclass/ds2_{train,valid}.csv`
- `src/linearclass/logreg.py`
- `src/linearclass/gda.py`

每个文件包含 $n$ 个样本，每行一个样本 $(x^{(i)}, y^{(i)})$. 具体而言，第 $i$ 行包含 $x_1^{(i)} \in \mathbb{R}, x_2^{(i)} \in \mathbb{R}$ 和 $y^{(i)} \in \{0,1\}$ 三列。在接下来的子问题中，我们将研究如何使用逻辑回归和高斯判别分析（GDA）对这两个数据集进行二元分类。

(a) \[10 分\]

课程中我们学习了逻辑回归的平均经验损失：

$$
J(\theta) = -\frac{1}{n} \sum_{i=1}^n \left( y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right),
$$

其中 $y^{(i)} \in \{0,1\}$, $h_\theta(x) = g(\theta^T x)$, 且 $g(z) = 1/(1 + e^{-z})$.

求该函数的 Hessian 矩阵 $H$, 并证明对于任意向量 $z$, 均有 $z^T H z \geq 0$.

**提示**：可先证明 $\sum_i \sum_j z_i x_i x_j z_j = (x^T z)^2 \geq 0$. 同时注意 $g'(z) = g(z)(1 - g(z))$.

**备注**：这是证明矩阵 $H$ 半正定 (记为 $H \succeq 0$) 的标准方法之一。由此可推出 $J$ 是凸函数，且除全局最小值外不存在局部最小值。若采用其他方法证明 $H \succeq 0$, 也可使用。

(b) \[5 分\] 编程题

根据 `src/linearclass/logreg.py` 中的指导，使用牛顿法训练逻辑回归分类器。从 $\theta = 0$ 开始运行牛顿法，直至 $\theta$ 的更新量足够小：具体而言，训练至第一次满足 $\|\theta_k - \theta_{k-1}\|_1 < \epsilon$ 的迭代 $k$, 其中 $\epsilon = 1 \times 10^{-5}$. 确保将模型在验证集上的预测概率写入代码指定的文件中。

绘制验证数据的散点图，横轴为 $x_1$, 纵轴为 $x_2$. 为区分两个类别，使用不同标记表示 $y^{(i)} = 0$ 和 $y^{(i)} = 1$ 的样本。在同一图中，绘制逻辑回归得到的决策边界 (即对应 $p(y|x) = 0.5$ 的直线)。

(c) \[5 分\] 回顾在高斯判别分析中，我们通过以下方程对 $(x, y)$ 的联合分布进行建模：

$$
\begin{aligned}
	p(y) &= \begin{cases}
		\phi & \text{若 } y = 1 \\
		1 - \phi & \text{若 } y = 0
	\end{cases} \\
	p(x|y=0) &= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_0)^T \Sigma^{-1} (x - \mu_0)\right) \\
	p(x|y=1) &= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_1)^T \Sigma^{-1} (x - \mu_1)\right),
\end{aligned}
$$

其中 $\phi, \mu_0, \mu_1$ 和 $\Sigma$ 是模型的参数。

假设我们已经拟合了参数 $\phi, \mu_0, \mu_1$ 和 $\Sigma$, 现在需要给定新数据点 $x$ 预测 $y$. 为了证明 GDA 得到的分类器具有线性决策边界，请证明后验分布可以表示为：

$$
p(y=1 \mid x; \phi, \mu_0, \mu_1, \Sigma) = \frac{1}{1 + \exp(-(\theta^T x + \theta_0))},
$$

其中 $\theta \in \mathbb{R}^d$ 和 $\theta_0 \in \mathbb{R}$ 是 $\phi, \mu_0, \mu_1$ 和 $\Sigma$ 的适当函数。

(d) \[7 分\] 对于给定数据集，我们声称参数的最大似然估计由以下公式给出：

$$
\begin{aligned}
	\hat{\phi} &= \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\} \\
	\hat{\mu}_0 &= \frac{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 0\}} \\
	\hat{\mu}_1 &= \frac{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\}} \\
	\hat{\Sigma} &= \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
\end{aligned}
$$

数据的对数似然函数为：

$$
\begin{aligned}
	\ell(\phi, \mu_0, \mu_1, \Sigma) 
		&= \log \prod_{i=1}^n p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) \\
		&= \log \prod_{i=1}^n p(x^{(i)} | y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi).
\end{aligned}
$$

通过最大化 $\ell$ 对四个参数的取值，证明 $\phi, \mu_0, \mu_1$ 和 $\Sigma$ 的最大似然估计确实由以上公式给出。(可假设数据集中至少存在一个正例和一个负例，以确保 $\mu_0$ 和 $\mu_1$ 定义中的分母非零。)

(e) \[5 分\] 编程题

在 `src/linearclass/gda.py` 中填写代码以计算 $\phi, \mu_0, \mu_1$ 和 $\Sigma$, 使用这些参数推导 $\theta$, 并利用得到的 GDA 模型对验证集进行预测。确保将模型在验证集上的预测结果写入代码指定的文件中。

绘制**验证数据**的散点图，横轴为 $x_1$, 纵轴为 $x_2$. 为区分两个类别，使用不同标记表示 $y^{(i)} = 0$ 和 $y^{(i)} = 1$ 的样本。在同一图中，绘制 GDA 得到的决策边界 (即对应 $p(y|x) = 0.5$ 的直线)。

(f) \[2 分\] 对于数据集 1，比较在部分 (b) 和部分 (e) 中分别通过逻辑回归和 GDA 得到的验证集图，并用几句话简要评述你的观察结果。

(g) \[5 分\] 对数据集 2 重复部分 (b) 和部分 (e) 的步骤。在数据集 2 的**验证集**上创建类似的图，并将这些图包含在你的报告中。

在哪个数据集上 GDA 的表现似乎比逻辑回归差？这可能是什么原因？

(h) \[1 分\] 对于在 (f) 和 (g) 中 GDA 表现较差的数据集，你能否找到一种对 $x^{(i)}$ 的变换，使得 GDA 的表现显著改善？这种变换可能是什么？

> [!example]- 答案  
>   ![[CS229_CN/Problem_Set/problem_set_1/solution#1. 线性分类器 (逻辑回归与高斯判别分析)]]

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