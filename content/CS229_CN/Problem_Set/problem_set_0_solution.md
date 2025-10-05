---
title: "习题集 #0 解答"
---
### 1. 梯度与 Hessians

(a) $\nabla f(x) = Ax + b$

(b) $\nabla f(x) = g'(h(x)) \odot \nabla h(x)$

(c) $\nabla^2 f(x) = A$

(d) $\nabla f(x) = g'(a^Tx) \ a$, $\nabla^2 f(x) = g''(a^Tx) \ aa^T$

### 2. 正定矩阵

(a) 有 1. $A^T=(zz^T)^T=zz^T=A$ 和 2. $x^TAx=x^Tzz^Tx=(z^Tx)^T(z^Tx)=\|z^Tx\|_2^2 \ge 0, \forall x > 0$; 所以 $A$ 为半正定矩阵。 

(b) $N(A) = \{x \in \mathbb{R}^n: x^Tz=0\}$, $\operatorname{rank}(A)=\operatorname{rank}(zz^T)=1$. 

(c) 是。有 1. $(BAB^T)^T=BA^TB^T=BAB^T$ 和 2. $x^TBAB^Tx=(B^Tx)A(B^Tx)= \ge 0, \forall x > 0$; 所以 $BAB^T$ 为半正定矩阵。

### 3. 特征向量、特征值与谱定理

(a) $[At^{(1)},\cdots,At^{(n)}]=AT=T\Lambda T^{-1}T=T\Lambda = [\lambda_1t^{(1)}, \cdots, \lambda_1t^{(n)}]$

(b) $[Au^{(1)},\cdots,Au^{(n)}]=AU=U\Lambda U^TU=U\Lambda = [\lambda_1u^{(1)}, \cdots, \lambda_1u^{(n)}]$

(c) $A t^{(i)} = \lambda_i t^{(i)}$, ${t^{(i)}}^TA t^{(i)} = \lambda_i \|t^{(i)}\|2^2 \ge 0$, 所以对每个 $i$, 都有 $\lambda_i(A) \geq 0$.