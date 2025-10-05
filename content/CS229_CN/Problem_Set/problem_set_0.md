---
title: "习题集 #0: 线性代数与多元微积分"
---
说明：(1) 本作业问题需深入思考，但无需长篇作答，请力求简洁。(2) 如有疑问，欢迎在课程 [Piazza 论坛](https://piazza.com/stanford/fall2018/cs229)发帖讨论。(3) 若未参加首次课程或对合作规范与学术诚信条款存在疑问，请在开始作答前查阅课程网站提供的《手册#1》中的相关政策。(4) 本次作业*不计分*，但建议完成所有习题以巩固线性代数知识，部分内容可能适用于后续作业。此外，本作业将作为 Gradescope 提交系统的使用导引。

---

### 1. \[0 分\] 梯度与 Hessians

若矩阵$A \in \mathbb{R}^{n \times n}$ 满足$A^T = A$, 即对所有$i, j$ 有$A_{ij} = A_{ji}$, 则称其为*对称*矩阵。同时回顾函数$f: \mathbb{R}^n \to \mathbb{R}$ 的梯度$\nabla f(x)$, 即由偏导数组成的$n$ 维向量：

$$
\nabla f(x) = \begin{bmatrix} \frac{\partial}{\partial x_1} f(x) \\ \vdots \\ \frac{\partial}{\partial x_n} f(x) \end{bmatrix} \quad \text{其中} \quad x = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}.
$$

函数$f: \mathbb{R}^n \to \mathbb{R}$ 的黑塞矩阵$\nabla^2 f(x)$ 是由二阶偏导数构成的$n \times n$ 对称矩阵：

$$
\nabla^2 f(x) = \begin{bmatrix}
\frac{\partial^2}{\partial x_1^2} f(x) & \frac{\partial^2}{\partial x_1 \partial x_2} f(x) & \cdots & \frac{\partial^2}{\partial x_1 \partial x_n} f(x) \\
\frac{\partial^2}{\partial x_2 \partial x_1} f(x) & \frac{\partial^2}{\partial x_2^2} f(x) & \cdots & \frac{\partial^2}{\partial x_2 \partial x_n} f(x) \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2}{\partial x_n \partial x_1} f(x) & \frac{\partial^2}{\partial x_n \partial x_2} f(x) & \cdots & \frac{\partial^2}{\partial x_n^2} f(x)
\end{bmatrix}.
$$

(a) 设$f(x) = \frac{1}{2} x^T A x + b^T x$, 其中$A$ 为对称矩阵，$b \in \mathbb{R}^n$ 为向量。试求$\nabla f(x)$？

(b) 设$f(x) = g(h(x))$, 其中$g: \mathbb{R} \to \mathbb{R}$ 与$h: \mathbb{R}^n \to \mathbb{R}$ 均为可微函数。试求$\nabla f(x)$？

(c) 设$f(x) = \frac{1}{2} x^T A x + b^T x$, 其中$A$ 为对称矩阵，$b \in \mathbb{R}^n$ 为向量。试求$\nabla^2 f(x)$？

(d) 设$f(x) = g(a^T x)$, 其中$g: \mathbb{R} \to \mathbb{R}$ 连续可微，$a \in \mathbb{R}^n$ 为向量。试求$\nabla f(x)$ 与$\nabla^2 f(x)$？(提示：$\nabla^2 f(x)$ 的表达式可简化为仅含 11 个符号，包括$\nabla$ 与括号)

---

### 2. \[0分\] 正定矩阵

若矩阵 $A \in \mathbb{R}^{n \times n}$ 满足 $A = A^T$ 且对所有 $x \in \mathbb{R}^n$ 有 $x^T A x \geq 0$, 则称其为*半正定*矩阵，记作 $A \succeq 0$. 若矩阵 $A$ 满足 $A = A^T$ 且对所有非零向量 $x \neq 0$ 有 $x^T A x > 0$, 则称其为*正定*矩阵，记作 $A \succ 0$. 正定矩阵的最简范例是单位矩阵 $I$ (对角线元素为 $1$、非对角线元素为 $0$ 的对角矩阵)，其满足 $x^T I x = \|x\|_2^2 = \sum_{i=1}^n x_i^2$.

(a) 设 $z \in \mathbb{R}^n$ 为一 $n$ 维向量。证明 $A = z z^T$ 是半正定矩阵。

(b) 设 $z \in \mathbb{R}^n$ 为一*非零* $n$ 维向量，且 $A = z z^T$. 问：$A$ 的零空间是什么？$A$ 的秩是多少？

(c) 设 $A \in \mathbb{R}^{n \times n}$ 为半正定矩阵，$B \in \mathbb{R}^{m \times n}$ 为任意矩阵，其中 $m, n \in \mathbb{N}$. 问：$B A B^T$ 是否为半正定矩阵？若是，请予以证明；若否，请给出具体的 $A$ 与 $B$ 作为反例。

---

### 3. \[0分\] 特征向量、特征值与谱定理

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