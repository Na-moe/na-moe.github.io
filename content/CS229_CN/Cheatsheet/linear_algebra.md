---
title: 线性代数速查表
---
## 通用符号

### 向量

我们记 $\boldsymbol{x} \in \mathbb{R}^n$ 为一个 $n$ 维向量，其中 $\boldsymbol{x} \in \mathbb{R}$ 是第 $i$ 维：

$$
\boldsymbol{x} = \begin{pmatrix}
	\ x_1 \ \\
	\ x_2 \ \\
	\ \vdots \ \\
	\ x_n \
\end{pmatrix} \in \mathbb{R}^n
$$

### 矩阵 

我们记 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ 为一个 $m$ 行 $n$ 列的矩阵，其中 $A_{i,j}$ 是第 $i$ 行 $j$ 列的元素：

$$
\boldsymbol{A} = \begin{pmatrix}
	\ A_{1,1} & \cdots & A_{1,n} \ \\
	\ \vdots & & \vdots \ \\
	\ A_{m,1} & \cdots & A_{m,n} \
\end{pmatrix} \in \mathbb{R}^{m \times n}
$$

*注: 上述定义的向量 $\boldsymbol{x}$ 也可视为一个 $n \times 1$ 维的矩阵，也常被被称为列向量。*

### 单位矩阵

单位矩阵 $\boldsymbol{I} \in \mathbb{R}^{n\times n}$ 是一个方阵，其对角线上元素为 $1$, 其余元素均为 $0$ :

$$
\boldsymbol{I} = \begin{pmatrix}
	\ 1 & 0 & \cdots & 0 \ \\
	\ 0 & \ddots & \ddots & \vdots \ \\
	\ \vdots & \ddots & \ddots & 0 \ \\
	\ 0 & \cdots & 0 & 1 \ 
\end{pmatrix}
$$

*注: 对于任意矩阵 $\boldsymbol{A}$, 有 $\boldsymbol{A}\boldsymbol{I}=\boldsymbol{I}\boldsymbol{A}=\boldsymbol{A}$.*

### 对角矩阵

对角矩阵 $\boldsymbol{D} \in \mathbb{R}^{n\times n}$ 是一个方阵，其对角线上元素不为 $0$, 其余元素均为 $0$ :

$$
\boldsymbol{A} = \begin{pmatrix}
	\ d_1 & 0 & \cdots & 0 \ \\
	\ 0 & \ddots & \ddots & \vdots \ \\
	\ \vdots & \ddots & \ddots & 0 \ \\
	\ 0 & \cdots & 0 & d_n \ 
\end{pmatrix}
$$

*注: 也记 $\boldsymbol{D}$ 为 $\mathrm{diag}(d_1, \dots, d_N)$.*

## 矩阵运算

### 向量-向量

存在两种向量-向量乘法：

* 内积: 对于 $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n$, 我们有： 

$$
\boxed{\boldsymbol{x} \boldsymbol{y}^T = \sum_{i=1}^n {x_i y_i} \in \mathbb{R}}
$$

* 外积: 对于 $\boldsymbol{x} \in \mathbb{R}^m, \boldsymbol{y} \in \mathbb{R}^n$, 我们有：

$$
\boxed{
	\boldsymbol{x}^T \boldsymbol{y} = \begin{pmatrix}
		\ x_1 y_1 & \cdots & x_1 y_n \ \\
		\ \vdots &  & \vdots \ \\
		\ x_m y_1 & \cdots & x_m y_n \ \\
	\end{pmatrix} \in \mathbb{R}^{m \times n}
}
$$

### 矩阵-向量

矩阵 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ 和向量 $\boldsymbol{x} \in \mathbb{R}^n$ 的乘积是一个大小为 $\mathbb{R}^m$ 的向量，满足：

$$
\boxed{
	\boldsymbol{A} \boldsymbol{x} 
	 = \begin{pmatrix}
			\ \boldsymbol{a}_{1,:}^T \boldsymbol{x} \ \\
			\ \vdots \ \\
			\ \boldsymbol{a}_{m,:}^T \boldsymbol{x} \
		\end{pmatrix}
	 = \sum_{i=1}^n {\boldsymbol{a}_{:,i} \boldsymbol{x}_i} \in \mathbb{R}^m
 }
$$

其中 $\boldsymbol{a}_{i,:}^T, \boldsymbol{a}_{:,i}$ 分别是 $\boldsymbol{A}$ 的行、列向量，$x_i$ 是 $\boldsymbol{x}$ 的元素。

### 矩阵-矩阵

矩阵 $\boldsymbol{A} \in \mathbb{R}^{m \times p}$ 和矩阵 $\boldsymbol{B} \in \mathbb{R}^{p \times n}$ 的乘积是一个大小为 $\mathbb{R}^{m \times n}$ 的矩阵，满足：

$$
\boxed{
	\boldsymbol{A} \boldsymbol{x} 
	 =  \begin{pmatrix}
			\ \boldsymbol{a}_{1,:}^T \boldsymbol{b}_{:,1} & \cdots & \boldsymbol{a}_{1,:}^T \boldsymbol{b}_{:,n} \ \\
			\ \vdots & & \vdots \ \\
			\ \boldsymbol{a}_{m,:}^T \boldsymbol{b}_{:,1} & \cdots & \boldsymbol{a}_{m,:}^T \boldsymbol{b}_{:,n} \
		\end{pmatrix}
	 = \sum_{i=1}^n {\boldsymbol{a}_{:,i} \boldsymbol{b}_{i,:}^T} \in \mathbb{R}^{m \times n}
 }
$$

其中 $\boldsymbol{a}_{i,:}^T, \boldsymbol{a}_{:,i}$ 分别是 $\boldsymbol{A}$ 的行、列向量，$\boldsymbol{b}_{i,:}^T, \boldsymbol{b}_{:,i}$ 分别是 $\boldsymbol{B}$ 的行、列向量。

### 转置

矩阵 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$ 的转置，记作 $\boldsymbol{A}^T$, 是其中元素沿对角线反转得到的：

$$
\boxed{
	\forall i,j, \quad \boldsymbol{A}_{i,j}^T = \boldsymbol{A}_{j,i}
}
$$

*注: 对于矩阵 $\boldsymbol{A}, \boldsymbol{B}$, 有 $(\boldsymbol{A}\boldsymbol{B})^T = \boldsymbol{B}^T\boldsymbol{A}^T$.*

### 逆

可逆方阵 $A$ 的逆记作 $A^{-1}$, 是唯一满足下列要求的矩阵：

$$
\boxed{
	\boldsymbol{A}\boldsymbol{A}^{-1}=\boldsymbol{A}^{-1}\boldsymbol{A}=\boldsymbol{I}
}
$$

*注: 不是所有方阵都是可逆的。同样的，对于矩阵 $\boldsymbol{A}, \boldsymbol{B}$, 有 $(\boldsymbol{A}\boldsymbol{B})^{-1} = \boldsymbol{B}^{-1}\boldsymbol{A}^{-1}$.*

### 迹

方阵 $\boldsymbol{A}$ 的迹，记作 $\mathrm{tr}(\boldsymbol{A})$, 是其对角线元素的和：

$$
\boxed{
	\mathrm{tr}(\boldsymbol{A}) = \sum_{i=1}^n A_{i,i}
}
$$

*注: 对于矩阵 $\boldsymbol{A}, \boldsymbol{B}$, 有 $\mathrm{tr}(\boldsymbol{A}^T) = \mathrm{tr}(\boldsymbol{A})$ 和 $\mathrm{tr}(\boldsymbol{A}\boldsymbol{B}) = \mathrm{tr}(\boldsymbol{B}\boldsymbol{A})$.*

### 行列式

方阵的行列式，记作 $|\boldsymbol{A}|$ 或者 $\det{\boldsymbol{A}}$, 可以用去掉其第 $i$ 行第 $j$ 列的矩阵 $\boldsymbol{A}_{\backslash i, \backslash j}$ 递归表示：

$$
\boxed{
	\det{\boldsymbol{A}} = |\boldsymbol{A}| = 
		\begin{cases}
		A_{1,1}, & \text{if } \boldsymbol{A} \in \mathbb{R}^{1\times 1} \\
		\sum_{j=1}^n {(-1)^{i+j} A_{i,j} |\boldsymbol{A}_{\backslash i, \backslash j}|}, & \text{if } \boldsymbol{A} \in \mathbb{R}^{n\times n}, \ n \ne 1 \\
		\end{cases}
}
$$

*注: $\boldsymbol{A}$ 可逆当且仅当 $|\boldsymbol{A}| \ne 0$. 同样，有 $|\boldsymbol{A}\boldsymbol{B}|=|\boldsymbol{A}||\boldsymbol{B}|$ 和 $\boldsymbol{A^T}=\boldsymbol{A}$.*

## 矩阵的性质

### 对称分解

对于给定矩阵 $\boldsymbol{A}$, 可以用其对称部分和反对称部分表示：

$$
\boxed{
	\boldsymbol{A} = \underbrace{\frac{\boldsymbol{A}+\boldsymbol{A}^T}{2}}_{\text{对称部分}} + \underbrace{\frac{\boldsymbol{A}-\boldsymbol{A}^T}{2}}_{\text{反对称部分}}
}
$$

### 范数

范数是一个函数 $\mathcal{N}: \mathbb{V} \to [0, +\infty)$, 其中 $\mathbb{V}$ 是一个向量空间, 并且对于所有 $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{V}$,  有：

* $\mathcal{N}(\boldsymbol{x}+\boldsymbol{y}) \le \mathcal{N}(\boldsymbol{x}) + \mathcal{N}(\boldsymbol{y})$
* 对于标量 $a$, 有 $\mathcal{N}(a\boldsymbol{x}) = |a| \mathcal{N}(\boldsymbol{x})$
* 如果 $\mathcal{N}(\boldsymbol{x})=0$, 则 $\boldsymbol{x}=0$

对于 $\boldsymbol{x} \in \mathbb{V}$, 下表总结了最常用的范数：


|         范数         |             符号              |                       定义                        |    用例    |
| :----------------: | :-------------------------: | :---------------------------------------------: | :------: |
|   曼哈顿范数，$\ell_1$   |   $\|\boldsymbol{x}\|_1$    |       $\sum_{i=1}^n \left\| x_i \right\|$       | LASSO 回归 |
|  欧几里得范数，$\ell_2$   |   $\|\boldsymbol{x}\|_2$    |          $\sqrt{\sum_{i=1}^n {x_i^2}}$          |   岭回归    |
|  $p$-范数，$\ell_p$   |   $\|\boldsymbol{x}\|_p$    | ${\left(\sum_{i=1}^n {x_i^p}\right)}^{\frac1p}$ |  霍尔德不等式  |
| 无穷范数，$\ell_\infty$ | $\|\boldsymbol{x}\|_\infty$ |                $\max_i \|x_i\|$                 |   一致收敛   |

### 线性相关

若一个向量集合中的一个向量可以由集合中其他向量的线性组合表示，则这个向量集合是线性相关的。

*注: 若其中任意向量都不能这么表示，则这个向量集合是线性无关的。*

### 矩阵的秩

给定矩阵 $\boldsymbol{A}$ 的秩记作 $\mathrm{rank}(\boldsymbol{A})$, 是由其列向量张成的向量空间的维度。这等价于 $\boldsymbol{A}$ 的线性无关列向量的最大数目。

### 半正定矩阵

如果矩阵 $\boldsymbol{A}$ 满足下式，则称其为半正定矩阵，记作 $\boldsymbol{A} \succeq 0$ :

$$
\boxed {\boldsymbol{A} = \boldsymbol{A}^T} 
\quad \text{且} \quad
\boxed{\forall \boldsymbol{x} \in \mathbb{R}^n, \boldsymbol{x}^T\boldsymbol{A}\boldsymbol{x} \ge 0}
$$
*注: 同样，如果一个矩阵 $\boldsymbol{A}$ 是 半正定矩阵，并且对于所有非零向量 $\boldsymbol{x}$ ，都满足 $\boldsymbol{x}^T\boldsymbol{A}\boldsymbol{x} \gt 0$, 那么这个矩阵 $\boldsymbol{A}$ 被称为正定矩阵，记作 $\boldsymbol{A} \succ 0$ .*

### 特征值，特征向量

对于给定矩阵 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$, 如果存在一个向量 $\boldsymbol{z} \in \mathbb{R}^n \setminus \{0\}$ 满足下式，则 $\lambda$ 被称为 $\boldsymbol{A}$ 的一个特征值：

$$
\boxed{
	\boldsymbol{A}\boldsymbol{z} = \lambda\boldsymbol{z}
}
$$

### 谱定理

对于矩阵 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$, 如果 $\boldsymbol{A}$ 可以被一个实正交矩阵 $\boldsymbol{U} \in \mathbb{R}^{n \times n}$ 对角化，则称 $\boldsymbol{A}$ 是对称的。记 $\boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1, \cdots, \lambda_n)$, 有：

$$
\boxed{
	\exists \ \text{对角阵 } \boldsymbol{\Lambda}, \quad
	\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^T
}
$$

### 奇异值分解

对于矩阵 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, 奇异值分解 (SVD) 是一个因子分解技巧，能保证存在酉矩阵 $\boldsymbol{U} \in \mathbb{R}^{m \times m}, \boldsymbol{V} \in \mathbb{R}^{n \times n}$ 和对角矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$, 满足：

$$
\boxed{
	\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T
}
$$

## 矩阵的微积分

### 梯度

令 $f: \mathbb{R}^{m \times n} \to \mathbb{R}$ 为一个函数，$\boldsymbol{A} \in \mathbb{R}^{m \times n}$ 为一个矩阵。$f$ 关于 $\boldsymbol{A}$ 的梯度是一个 $m \times n$ 的矩阵，记作 $\nabla_{\boldsymbol{A}} f(\boldsymbol{A})$, 满足：

$$
\boxed{
	\left( \nabla_{\boldsymbol{A}} f(\boldsymbol{A}) \right)_{i,j} = \frac{\partial f(\boldsymbol{A})}{\partial A_{i,j}}
}
$$

### Hessian

令 $f: \mathbb{R}^{m \times n} \to \mathbb{R}$ 为一个函数，$\boldsymbol{x} \in \mathbb{R}^{n}$ 为一个向量。$f$ 关于 $\boldsymbol{x}$ 的 Hessian 是一个 $n \times n$ 的对称矩阵，记作 $\nabla_{\boldsymbol{x}}^2 f(\boldsymbol{\boldsymbol{x}})$, 满足：

$$
\boxed{
	\left( \nabla_{\boldsymbol{x}}^2 f(\boldsymbol{\boldsymbol{x}}) \right)_{i,j} = \frac{\partial f(\boldsymbol{x})}{\partial x_{i}\partial x_{j}}
}
$$

### 梯度运算

对于矩阵 $A, B, C$, 下列梯度性质值得记住：

$$
\begin{gather*}
	\boxed{
		\nabla_{\boldsymbol{A}} \mathrm{tr}(\boldsymbol{A}\boldsymbol{B})  = \boldsymbol{B}^T
	} \quad
	\boxed{
		\nabla_{\boldsymbol{A}}^T f(\boldsymbol{A}) = \left( \nabla_{\boldsymbol{A}} f(\boldsymbol{A}) \right)^T
	} \\
	\boxed{
		\nabla_{\boldsymbol{A}} \mathrm{tr}(\boldsymbol{A}\boldsymbol{B}\boldsymbol{A}^T\boldsymbol{C})  = \boldsymbol{C}\boldsymbol{A}\boldsymbol{B} + \boldsymbol{C}^T\boldsymbol{A}\boldsymbol{B}^T
	} \quad
	\boxed{
		\nabla_{\boldsymbol{A}} \deg{A} = \left(\deg{A} \right)\left( \boldsymbol{A}^{-1} \right)^T
	}
\end{gather*}
$$

## 译注

仍然有很多重要的线性代数概念在此速查表中没有提及，笔者认为最重要的一点是从线性空间的角度理解矩阵运算，关于这一点的图解可以参见[线性代数的艺术](https://github.com/kf-liu/The-Art-of-Linear-Algebra-zh-CN/blob/main/The-Art-of-Linear-Algebra-zh-CN.pdf)；对于矩阵的梯度运算，笔者认为[矩阵求导术](https://zhuanlan.zhihu.com/p/24709748)一文授人以渔，从整体出发建立了矩阵求导的技术，并给出了详细的示例，十分值得一读。