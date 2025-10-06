---
title: "习题集 #1: 答案"
---
### 1. 线性分类器 (逻辑回归与高斯判别分析)

(a) \[10 分\]

记 $z_i=\theta^Tx^{(i)}$, 有 $\frac{\mathrm{d}z_i}{\mathrm{d}\theta_j}=x^{(i)}_j$, 先考虑单个样本的损失 $\ell^{(i)}(\theta)$:

$$
\ell^{(i)}(\theta) = - \left[ y^{(i)} \log g(z_i) + (1-y^{(i)}) \log(1-g(z_i)) \right],
$$

对 $\ell^{(i)}(\theta)$ 关于 $\theta_j$ 求导:

$$
\begin{aligned}
	\frac{\partial{\ell^{(i)}}}{\partial{\theta_j}}
		&= -\left[y^{(i)}\frac{g'(z_i)}{g(z_i)}\frac{\mathrm{d}z_i}{\mathrm{d}\theta_j} + (1-y^{(i)})\frac{-g'(z_i)}{1-g(z_i)}\frac{\mathrm{d}z_i}{\mathrm{d}\theta_j}\right] \\
		&= -\left[y^{(i)} \frac{g(z_i)(1-g(z_i))}{g(z_i)} x^{(i)}_j + (1-y^{(i)})\frac{-g(z_i)(1-g(z_i))}{1-g(z_i)} x^{(i)}_j \right] \\
		&= - \left[y^{(i)}(1-g(z_i)) -g(z_i)(1-y^{(i)}) \right] x^{(i)}_j \\
		&= (g(z_i) - y^{(i)})x^{(i)}_j
\end{aligned}
$$

再关于 $\theta_k$ 求导:

$$
\begin{aligned}
	\frac{\partial^2{\ell^{(i)}}}{\partial{\theta_j} \partial{\theta_k}}
		&= \frac{\partial(g(z_i) - y^{(i)})x^{(i)}_j}{\partial{\theta_k}} \\
		&= \frac{\partial{g(z_i)}}{\partial{\theta_k}}x^{(i)}_j \\
		&= h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) x^{(i)}_j x^{(i)}_k
\end{aligned}
$$

即 $H_{j,k} = \frac1n\sum_{i=1}^n h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) x^{(i)}_j x^{(i)}_k$, 所以有:

$$
\boxed{H = \frac1n\sum_{i=1}^n h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) x^{(i)} {x^{(i)}}^T}
$$

又因为 $0 < h_\theta < 1$, 所以对于任意 $z$, 有:

$$
\begin{aligned}
	z^THz 
		&= \frac1n\sum_{i=1}^n h_\theta(x^{(i)}) (1-h_\theta(x^{(i)})) z^T x x^T z \\
		&= \frac1n\sum_{i=1}^n \underbrace{h_\theta(x^{(i)})}_{>0}\ \underbrace{(1-h_\theta(x^{(i)}))}_{>0} \ \underbrace{\|z^Tx\|_2^2}_{\ge0} \ge 0
\end{aligned}
$$

(b) \[5 分\]

(c) \[5 分\]

$$
\begin{aligned}
	p(y=1 \mid x) 
		&= \frac{p(x \mid y=1) p(y=1)}{p(x \mid y=0) p(y=0)+p(x \mid y=1) p(y=1)} \\
		&= \frac{\frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_1)^T \Sigma^{-1} (x - \mu_1)\right) \phi}{\frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_0)^T \Sigma^{-1} (x - \mu_0)\right) (1-\phi) + \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_1)^T \Sigma^{-1} (x - \mu_1)\right) \phi} \\
		&= \frac1{1+\frac{1-\phi}{\phi} \exp\left(\frac{1}{2}(x - \mu_1)^T \Sigma^{-1} (x - \mu_1) -\frac{1}{2}(x - \mu_0)^T \Sigma^{-1} (x - \mu_0)\right)} \\
		&= \frac1{1+ \exp\left(-\left( \underbrace{(\mu_1 - \mu_0)^T\Sigma^{-1}}_{\theta^T}x + \underbrace{\frac12  (\mu_0^T\Sigma^{-1}\mu_0 - \mu_1^T\Sigma^{-1}\mu_1) + \log{(\frac{\phi}{1-\phi}})}_{\theta_0} \right) \right)}
\end{aligned}
$$

即

$$
\boxed{
	p(y=1 \mid x) = \frac{1}{1+\exp{(-(\theta^Tx+\theta_0))}}
}
$$

其中 $\theta=\Sigma^{-1}(\mu_1 - \mu_0), \theta_0=\frac12  (\mu_0^T\Sigma^{-1}\mu_0 - \mu_1^T\Sigma^{-1}\mu_1) + \log{(\frac{\phi}{1-\phi})}$.