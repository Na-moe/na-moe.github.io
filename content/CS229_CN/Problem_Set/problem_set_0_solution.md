---
title: "习题集 #0: 答案"
---
### 1. 梯度与 Hessians

(a) $\nabla f(x) = Ax + b$

(b) $\nabla f(x) = g'(h(x)) \odot \nabla h(x)$

(c) $\nabla^2 f(x) = A$

(d) $\nabla f(x) = g'(a^Tx) \ a$, $\nabla^2 f(x) = g''(a^Tx) \ aa^T$

### 2. 正定矩阵

(a) 有 1. $A^T=(zz^T)^T=zz^T=A$ 和 2. $x^TAx=x^Tzz^Tx=(z^Tx)^T(z^Tx)=\|z^Tx\|_2^2 \ge 0, \forall x \in \mathbb{R}^n$; 所以 $A$ 为半正定矩阵。 

(b) $N(A) = \{x \in \mathbb{R}^n: x^Tz=0\}$, $\operatorname{rank}(A)=\operatorname{rank}(zz^T)=1$. 

(c) 是。有 1. $(BAB^T)^T=BA^TB^T=BAB^T$ 和 2. $x^TBAB^Tx=(B^Tx)A(B^Tx) \ge 0, \forall x \in \mathbb{R}^n$; 所以 $BAB^T$ 为半正定矩阵。

### 3. 特征向量、特征值与谱定理

(a) $[At^{(1)},\cdots,At^{(n)}]=AT=T\Lambda T^{-1}T=T\Lambda = [\lambda_1t^{(1)}, \cdots, \lambda_1t^{(n)}]$

(b) $[Au^{(1)},\cdots,Au^{(n)}]=AU=U\Lambda U^TU=U\Lambda = [\lambda_1u^{(1)}, \cdots, \lambda_1u^{(n)}]$

(c) $A t^{(i)} = \lambda_i t^{(i)}$, ${t^{(i)}}^TA t^{(i)} = \lambda_i \|t^{(i)}\|2^2 \ge 0$, 所以对每个 $i$, 都有 $\lambda_i(A) \geq 0$.

## 4. 概率论与多元高斯分布

(a) 

$\mathbb{E}(Y)=\mathbb{E}(X_1+X_2+\cdots+X_n)=\mathbb{E}(X_1)+\mathbb{E}(X_2)+\cdots+\mathbb{E}(X_n)=\mu_1+\mu_2+\cdots+\mu_n=\boldsymbol{1}^T\mu$,

$\mathrm{Var}(Y) = \mathrm{Var}\left( \sum_{i=1}^n X_i \right) = \sum_{i,j} \mathrm{Cov}(X_i, X_j)=\sum_{i,j}\sigma_{i,j}=\boldsymbol{1}^T\Sigma\boldsymbol{1}$.

因此 $Y$ 是一维正态分布：$Y \sim \mathcal{N}\left( \mathbf{1}^T \mu, \; \mathbf{1}^T \Sigma \mathbf{1} \right)$.

(b)

$$
\begin{aligned}
	\mathbb{E}[X^T \Sigma^{-1} X] 
		&= \mathbb{E}[\operatorname{tr}(X^T \Sigma^{-1} X)] \quad \text{(标量的迹是它本身)}\\
		&= \mathbb{E}[\operatorname{tr}(\Sigma^{-1} X X^T)] \quad \text{(迹中的矩阵乘法可轮换)} \\
		&= \operatorname{tr}(\Sigma^{-1} \mathbb{E}[X X^T]) \quad \text{(迹和期望可交换)} \\
		&= \operatorname{tr}(\Sigma^{-1} \left(\mathbb{E}[(X-\mu)(X-\mu)^T] + \mathbb{E}[X \mu^T + \mu X^T - \mu \mu^T]\right)) \\
		&= \operatorname{tr}(\Sigma^{-1} (\Sigma + \mu \mu^T)) \quad (\Sigma\text{ 的定义)}\\
		&= \operatorname{tr}(I_n+\Sigma^{-1}\mu \mu^T) \\
		&= n+\mu^T\Sigma^{-1}\mu
\end{aligned}
$$
