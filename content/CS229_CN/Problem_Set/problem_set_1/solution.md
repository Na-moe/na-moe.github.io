---
title: "习题集 #1: 答案"
---
### 1. 线性分类器 (逻辑回归与高斯判别分析)

(a)

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

(b)

```python
def fit(self, x, y):
	n, d = x.shape
	if self.theta is None:
		self.theta = np.zeros(d)

	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

	for i in range(self.max_iter):
		z = x.dot(self.theta)
		h = sigmoid(z)

		grad = (1 / n) * (x.T @ (h - y))
		hessian = (1 / n) * (x.T @ ((h * (1 - h))[:, np.newaxis] * x))
		theta_new = self.theta - np.linalg.inv(hessian) @ grad

		if np.linalg.norm(theta_new - self.theta, ord=1) < self.eps:
			break

		self.theta = theta_new
```

| 数据集 1                  | 数据集 2                  |
| ---------------------- | ---------------------- |
| ![[logreg_pred_1.svg]] | ![[logreg_pred_2.svg]] |

(c)

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

(d)

我们有

$$
\ell(\phi, \mu_0, \mu_1, \Sigma) = \sum_{i=1}^n \left[ \log p(x^{(i)} | y^{(i)}; \mu_0, \mu_1, \Sigma) + \log p(y^{(i)}; \phi) \right]
$$

先看含 $\phi$ 的部分 $\sum_{i=1}^n \left[ y^{(i)} \log \phi + (1-y^{(i)}) \log(1-\phi) \right]$, 对 $\phi$ 求导：

$$
\begin{aligned}
	&\frac{\partial \ell}{\partial \phi} = \frac{\sum_i y^{(i)}}{\phi} - \frac{\sum_i (1-y^{(i)})}{1-\phi} = 0 \\
	\Rightarrow \ &\frac{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\}}{\phi} = \frac{n-\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\}}{1-\phi} \\
	\Rightarrow \ &\boxed{\hat\phi = \frac1n \sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\}}
\end{aligned}
$$

再看高斯部分

$$
\begin{aligned}
	\sum_{i=1}^n \log p(x^{(i)}|y^{(i)}; \mu_0, \mu_1, \Sigma) = &-\frac{n d}{2} \log(2\pi) \\
		&- \frac{n}{2} \log|\Sigma| \\
		&- \frac12 \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})^T \Sigma^{-1} (x^{(i)} - \mu_{y^{(i)}})
\end{aligned}
$$

与 $\mu_0$ 有关的项为 $-\frac12 \sum_{y^{(i)}=0} (x^{(i)} - \mu_0)^T \Sigma^{-1} (x^{(i)} - \mu_0)$, 对 $\mu_0$ 求导：

$$
\begin{aligned}
	&\frac{\partial \ell}{\partial \mu_0} = \sum_{y^{(i)}=0} \Sigma^{-1} (x^{(i)} - \mu_0) = 0 \\
	\Rightarrow \ &\boxed{\hat\mu_0 = \sum_{y^{(i)}=0} x^{(i)} = \frac{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 0\}}}
\end{aligned}
$$

同理可得：

$$
\boxed{\hat\mu_1 = \frac{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^n \mathbf{1}\{y^{(i)} = 1\}}}
$$

令 $\Omega = \Sigma^{-1}$, 对 $\Omega$ 求导得

$$
\begin{aligned}
	&\frac{\partial \ell}{\partial \Omega} = \frac{n}{2} \Sigma - \frac12 \sum_i (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T = 0 \\
	\Rightarrow \ &\boxed{\hat{\Sigma} = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T}
\end{aligned}
$$

以上，所有最大似然估计得证。

(e)

```python
def fit(self, x, y):
	n, d = x.shape

	phi = np.mean(y)
	mu_0 = np.mean(x[y == 0], axis=0)
	mu_1 = np.mean(x[y == 1], axis=0)

	x_mu0 = (x - mu_0)[..., np.newaxis]
	x_mu0_T = x_mu0.transpose(0, 2, 1)
	x_mu1 = (x - mu_1)[..., np.newaxis]
	x_mu1_T = x_mu1.transpose(0, 2, 1)
	y_ = y[:, np.newaxis, np.newaxis]
	sigma = (x_mu0 @ x_mu0_T * (1 - y_) + x_mu1 @ x_mu1_T * y_).mean(axis=0)
	sigma_inv = np.linalg.inv(sigma)

	self.theta = np.zeros(d + 1)
	self.theta[1:] = sigma_inv @ (mu_1 - mu_0)
	self.theta[0] = 0.5 * (
		mu_0 @ sigma_inv @ mu_0 - mu_1 @ sigma_inv @ mu_1
	) + np.log(phi / (1 - phi))
```

| 数据集 1               | 数据集 2               |
| ------------------- | ------------------- |
| ![[gda_pred_1.svg]] | ![[gda_pred_2.svg]] |

(f)

对于数据集 1，逻辑回归在其验证集上的结果比 GDA 更好。

(g)

已包含在 (b) (e) 的解答中。

数据集 1 上 GDA 的表现更差，原因可能是数据集 1 的数据分布不满足 GDA 数据服从正态分布的假设。

(h)

使用 Box-cox 变换对数据集 1 进行处理，使其更接近正态分布。

### 2. 不完整的、仅含正类标签的数据

(a)

```python
x_train, y_train = util.load_dataset(train_path, label_col="t", add_intercept=True)

clf_a = LogisticRegression()
clf_a.fit(x_train, y_train)

x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
y_pred = clf_a.predict(x_test)
print(f"Part (a): Acc={np.mean((y_pred >= 0.5) == y_test):.3f}")

np.savetxt(output_path_true, y_pred)
util.plot(x_test, y_test, clf_a.theta, output_path_true.replace(".txt", ".svg"))
```

![[posonly_true_pred.svg|500]]

(b)

```python
x_train, y_train = util.load_dataset(train_path, label_col="y", add_intercept=True)

clf_b = LogisticRegression()
clf_b.fit(x_train, y_train)

x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
y_pred = clf_b.predict(x_test)
print(f"Part (b): Acc={np.mean((y_pred >= 0.5) == y_test):.3f}")

np.savetxt(output_path_naive, y_pred)
util.plot(x_test, y_test, clf_b.theta, output_path_naive.replace(".txt", ".svg"))
```

![[posonly_naive_pred.svg|500]]

(c)

$$
\begin{aligned}
	p(t^{(i)} = 1 \mid y^{(i)} = 1, x^{(i)}) &= \frac{p(y^{(i)} = 1 \mid t^{(i)} = 1, x^{(i)})\ p(t^{(i)} = 1 \mid x^{(i)})}{p(y^{(i)} = 1 \mid t^{(i)} = 1, x^{(i)})\ p(t^{(i)} = 1 \mid x^{(i)}) + p(y^{(i)} = 1 \mid t^{(i)} = 0, x^{(i)}) \ p(t^{(i)} = 0 \mid x^{(i)})} \\
	&= \frac{\alpha p(t^{(i)} = 1 \mid x^{(i)})}{\alpha p(t^{(i)} = 1 \mid x^{(i)})+0\cdot p(t^{(i)} = 0 \mid x^{(i)})} \\
	&=1
\end{aligned}
$$

(d)

$$
\begin{aligned}
	p(t^{(i)} = 1 \mid x^{(i)}) 
		&= \frac{1}{\alpha}\cdot (\alpha \ p(t^{(i)} = 1 \mid x^{(i)}) + 0 \cdot p(t^{(i)} = 0 \mid x^{(i)})) \\
		&= \frac{1}{\alpha}\cdot (p(y^{(i)} = 1 \mid t^{(i)} = 1, x^{(i)})\ p(t^{(i)} = 1 \mid x^{(i)}) + p(y^{(i)} = 1 \mid t^{(i)} = 0, x^{(i)}) \ p(t^{(i)} = 0 \mid x^{(i)})) \\
		&= \frac{1}{\alpha}\cdot p(y^{(i)} = 1 \mid x^{(i)})
\end{aligned}
$$

(e)

由 (d) 知

$$
p(y^{(i)} = 1 \mid x^{(i)}) = \alpha \cdot p(t^{(i)} = 1 \mid x^{(i)})
$$

现在利用关键假设：$p(t^{(i)} = 1 \mid x^{(i)}) \in \{0,1\}$, 又因为 $p(y^{(i)} = 1 \mid t^{(i)} = 0, x^{(i)}) = 0$, 所以如果 $y^{(i)} = 1$, 则必然 $t^{(i)} = 1$. 所以此时 $p(t^{(i)} = 1 \mid x^{(i)}) = 1$.

因此：

$$
p(y^{(i)} = 1 \mid x^{(i)}) = \alpha \cdot 1 = \alpha
$$

由于 $h(x^{(i)}) = p(y^{(i)} = 1 \mid x^{(i)})$，所以：
$$
\begin{aligned}
	&h(x^{(i)}) = \alpha \quad \text{当 } y^{(i)} = 1 \\
	\Rightarrow \ & \boxed{\mathbb{E}(h(x^{(i)}) \mid y^{(i)} = 1) = \alpha}
\end{aligned}
$$

(f)

```python
x_valid, y_valid = util.load_dataset(valid_path, label_col="y", add_intercept=True)
y_valid_pred = clf_b.predict(x_valid)

# Estimate alpha
alpha = np.mean(y_valid_pred[y_valid == 1])
print(f"Estimated alpha: {alpha:.3f}")

x_test, y_test = util.load_dataset(test_path, label_col="t", add_intercept=True)
y_pred = clf_b.predict(x_test) / alpha
y_pred = np.clip(y_pred, 0, 1)  # Ensure probabilities are in [0, 1]
print(f"Part (f): Acc={np.mean((y_pred >= 0.5) == y_test):.3f}")

np.savetxt(output_path_adjusted, y_pred)
util.plot(x_test, y_test, clf_b.theta, output_path_naive.replace(".txt", ".svg"), correction=alpha)
```

![[posonly_adjusted_pred.svg|500]]

### 3. 泊松回归

(a)

$$
\begin{aligned}
  p(y; \lambda) 
    &= \frac{e^{-\lambda} \lambda^y}{y!}, \\
    &= \underbrace{ \frac{1}{y!} }_{ b(y) } \cdot \exp(\underbrace{ y }_{ T(y) }\underbrace{ \log\lambda }_{ \eta }-\underbrace{ \lambda }_{ a(\eta)={e}^\eta })
\end{aligned}
$$

(b)

由 (a) 知，泊松分布的自然参数 $\eta = \log \lambda$, 泊松分布的均值 $\mu = \lambda$, 则典范响应函数为

$$
\begin{aligned}
  g(\eta) 
    &= E[T(y); \eta] \\
    &= \mu = \lambda \\
    &= \boxed{ {e}^\eta }
\end{aligned}
$$

(c)

由 (a), (b) 知，单个样本的对数似然为：

$$
\begin{aligned}
  \ell^{(i)} 
    &= \log p(y^{(i)} \mid x^{(i)}; \theta) \\
    &= \log \frac{e^{-\lambda^{(i)}} (\lambda^{(i)})^{y^{(i)}}}{y^{(i)}!} \\
    &= \log \frac{e^{-e^{\eta^{(i)}}} (e^{\eta^{(i)}})^{y^{(i)}}}{y^{(i)}!} \\
    &= \theta^Tx^{(i)}y^{(i)}- e^{\theta^Tx^{(i)}} - \log (y^{(i)}!)​
\end{aligned}
$$

对 $\theta$ 求导得：

$$
\begin{aligned}
  \frac{ \partial \ell^{(i)} }{ \partial \theta } 
    &= x^{(i)}y^{(i)} - e^{\theta^T x^{(i)}} x^{(i)} \\ 
    &= (y^{(i)} - e^{\theta^T x^{(i)}}) x^{(i)}
\end{aligned}
$$

所以随机梯度上升更新规则为：

$$
\begin{aligned}
  \theta 
    &:= \theta + \alpha \frac{ \partial \ell^{(i)} }{ \partial \theta } \\
    &\boxed{:= \theta + \alpha (y^{(i)} - e^{\theta^T x^{(i)}}) x^{(i)}}
\end{aligned}
$$

(d)

```python
	n, d = x.shape

	if self.theta is None:
		self.theta = np.zeros(d)

	for i in range(self.max_iter):
		eta = x @ self.theta
		mu = np.exp(eta)
		gradient = x.T @ (y - mu)
		update = self.step_size * gradient
		
		if np.linalg.norm(update) < self.eps:
			break
		self.theta += update
```

![[poisson_pred.svg|500]]

### 4. 广义线性模型的凸性

(a)

我们有 $\frac{ \partial p(y; \eta) }{ \partial \eta } = p(y; \eta) \left(T(y) - \frac{ \partial \alpha }{ \partial \eta }\right)$, 于是：

$$
\begin{gathered}
   &\int p(y; \eta) \, dy = 1 \\
   \underset{ \text{两侧对 } \eta \text{ 求导} }{ \Rightarrow } &\frac{ \partial  }{ \partial \eta } \int p(y; \eta) \, dy = 0 \\
   \underset{ 交换积分与求导 }{ \Rightarrow } \ & \int p(y; \eta) \left(T(y) - \frac{ \partial  }{ \partial \eta } \alpha(\eta)\right) \, dy = 0 \\
   \Rightarrow \ & \int T(y)p(y; \eta) \, dy - \frac{ \partial  }{ \partial \eta } \alpha(\eta)\int p(y; \eta) \, dy = 0 \\
   \Rightarrow \ & \mathbb{E}[T(y); \eta] - \frac{ \partial  }{ \partial \eta } \alpha(\eta) = 0 \\
   \Rightarrow \ & \boxed{\mathbb{E}[y; \eta] = \frac{ \partial  }{ \partial \eta } \alpha(\eta)}
\end{gathered}
$$

(b)

$$
\begin{gathered}
   &\int p(y; \eta) \, dy = 1 \\
   \underset{ \text{两侧对 } \eta \text{ 求二阶导} }{ \Rightarrow } &\frac{ \partial^2  }{ \partial \eta^2 } \int p(y; \eta) \, dy = 0 \\
   \underset{ 交换积分与求导 }{ \Rightarrow } \ & \int \left[ p(y; \eta) \left(T(y) - \frac{ \partial  }{ \partial \eta } \alpha(\eta)\right)^2 -  p(y; \eta) \frac{ \partial^2  }{ \partial \eta^2 }\alpha(\eta) \right] \, dy = 0 \\
   \Rightarrow \ & \int \left(T(y) - \frac{ \partial  }{ \partial \eta } \alpha(\eta)\right)^2p(y; \eta) \, dy - \frac{ \partial^2  }{ \partial \eta^2 } \alpha(\eta)\int p(y; \eta) \, dy = 0 \\
   \Rightarrow \ & \mathrm{Var}[T(y); \eta] - \frac{ \partial^2  }{ \partial \eta^2 } \alpha(\eta) = 0 \\
   \Rightarrow \ & \boxed{\mathrm{Var}[y; \eta] = \frac{ \partial^2  }{ \partial \eta^2 } \alpha(\eta)}
\end{gathered}
$$

(c)

$$
\begin{aligned}
  \ell(\theta) 
    &= -\log\left( \prod_{i} p(y^{(i)} \mid x^{(i)}; \theta) \right) \\
    &= -\sum_{i}\log \left( b(y^{(i)}) \exp({\eta^{(i)}}^T T(y^{(i)}) - a({\eta^{(i)}})) \right) \\
    &= -\sum_{i}\log \left( b(y^{(i)}) \exp({x^{(i)}}^T\theta T(y^{(i)}) - a(\theta^Tx^{(i)})) \right) \\
    &= -\sum_{i}\log b(y^{(i)}) - \sum_{i} ({x^{(i)}}^T\theta T(y^{(i)}) - a(\theta^Tx^{(i)}))
\end{aligned}
$$

对 $\theta$ 求导得：

$$
\begin{aligned}
  \nabla_{\theta}\ell
    &= -\sum_{i} \left( x^{(i)}T(y^{(i)}) -  \frac{ \partial  }{ \partial \eta^{(i)} } \alpha({\eta}^{(i)})x^{(i)} \right) \\
    &= \sum_{i} \left( \frac{ \partial  }{\partial \eta^{(i)}}  \alpha({\eta}^{(i)}) - T(y^{(i)}) \right) x^{(i)}
\end{aligned}
$$

再对 $\theta$ 求导得：

$$
\begin{aligned}
  H_{\theta} 
    &= -\sum_{i} \left( x^{(i)}T(y^{(i)}) -  \frac{ \partial  }{ \partial \eta^{(i)} } \alpha({\eta}^{(i)})x^{(i)} \right) \\
    &= \sum_{i} \frac{ \partial^2  }{\partial {\eta^{(i)}}^2}  \alpha({\eta}^{(i)}) x^{(i)}{x^{(i)}}^T \\
    &= \boxed{\sum_{i} \mathrm{Var}[y; \eta] x^{(i)}{x^{(i)}}^T}
\end{aligned}
$$

对于任意向量 $z$:

$$
\begin{gathered}
  &z^TH_{\theta}z = \sum_{i} \underbrace{ \mathrm{Var}[y; \eta] }_{ \ge 0 } \ \underbrace{ \|{x^{(i)}}^Tz\|^2_{2} }_{ \ge 0 } \ge 0 \\
  \Rightarrow \ & \boxed{H_{\theta} \text{ 正定}}
\end{gathered}
$$

所以 GLM 的 NLL 损失函数是**凸**的。

### 5. 线性回归：何为线性？

(a)

$$
\boxed{J(\theta) = \frac{1}{2} \sum_{i=1}^n {\|h_{\theta}(\hat{x}^{(i)}) - y^{(i)}\|}_{2}^2}
$$

$$
\begin{aligned}
  \theta &:= \theta - \alpha\nabla_{\theta}J \\
    &:= \boxed{\theta - \alpha \sum_{i=1}^n (h_{\theta}(\hat{x}^{(i)}) - y^{(i)})\hat{x}^{(i)}}
\end{aligned}

$$

(c)

```python
def fit(self, X, y):
	self.theta = np.linalg.solve(X.T.dot(X), X.T.dot(y))
	
def create_poly(self, k, X):
	n, d = X.shape
	assert d == 2, "Input should be of shape (n_examples, 2)"

	X_poly = np.ones((n, k + 1))
	for i in range(1, k + 1):
		X_poly[:, i] = X[:, 1] ** i
	return X_poly
```

![[lin_poly_3_train.svg|500]]

(d)

![[lin_poly_k_train.svg|500]]

随着 $k$ 的增加，曲线越来越贴近数据点，但是波动越来越大。

(e)

![[lin_sin_k_train.svg|500]]

小的 $k$ 也可以很好地拟合数据点。

(f)

![[lin_poly_k_small.svg|500]]

$k$ 较大时，拟合曲线的波动很大。