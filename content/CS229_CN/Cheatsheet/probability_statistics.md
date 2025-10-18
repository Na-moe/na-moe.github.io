---
title: 概率统计速查表
---
## 通用符号

|           |                            定义                            |                                                                           形式                                                                            |                           注释                           |
| --------- | :------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------: |
| **样本空间**  |                  一个实验所有可能结果的集合称为实验的样本空间                  |                                                                          $$S$$                                                                          | 向量 $\boldsymbol{x}$ 也可视为一个 $n \times 1$ 维的矩阵，也常被被称为列向量 |
| **事件**    |                     样本空间的任何子集被称为一个事件                     |                                                                   $$E \subseteq S $$                                                                    |            如果实验的结果包含在 $E$ 内，那么我们称作 $E$ 发生了             |
| **概率论公理** |                  记 $P(E)$ 为事件 $E$ 发生的概率                  | $$\begin{aligned}&(1)\quad 0 \le P(E) \le 1 \\ &(2)\quad P(S)=1 \\ &(3)\quad P\left(\bigcup_{i=1}^{n} E_{i} \right) = \sum_{i=1}^n P(E_{i}) \end{aligned}$$ |                                                        |
| **排列**    | 一个排列是从 $n$ 个对象的池子中按照给定次序安置 $r$ 个对象，这样的安置的数目用 $P(n,r)$ 表示 |                                                             $$P(n,r) = \frac{n!}{(n-r)!}$$                                                              |                                                        |
| **组合**    |  一个组合是从 $n$ 个对象的池子中无序地安置 $r$ 个对象，这样的安置的数目用 $C(n,r)$ 表示   |                                                  $$C(n,r) = \frac{P(n,r)}{r!} = \frac{n!}{r!(n-r)!}$$                                                   |      对于 $0 \le r \le n$, 我们有 $P(n,r) \ge C(n,r)$.      |

## 条件概率


|                   |                                               定义                                                |                                            形式                                            |                              注释                               |
| ----------------- | :---------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: | :-----------------------------------------------------------: |
| **贝叶斯定理**         |                                对于事件 $A$, $B$ 如果满足 $P(B) > 0$, 有                                 |                        $$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$                        | 我们有 $P(A\cap B)=P(A\mid B)P(B)$ 和 $P(A\mid B)=P(B\mid A)P(A)$ |
| **划分**            | 设 $\{A_{i},i\in[\![1,n]\!]\}$ 对所有 $i$ 满足 $A_i \neq \emptyset$. 称 $\{A_i\}$ 为样本空间 $S$ 上的一个划分，当满足 | $$\forall i\neq j,\ A_{i}\cap A_{j} = \emptyset, \text{ 且 } \bigcup_{i=1}^n A_{i} = S$$  |  对样本空间的任意事件 $B$, 我们有 $P(B)=\sum_{i=1}^n P(B\mid A_i) P(A_i)$  |
| **贝叶斯定理扩展形式**<br> |                          设 $\{A_{i},i\in[\![1,n]\!]\}$是样本空间上的一个划分，我们有                           | $$P(A_{k}\mid B)=\frac{P(B\mid A_{k})P(A_{k})}{\sum_{i=1}^nP(B\mid A_{{i}})P(A_{{i}})}$$ |                                                               |
| **独立**            |                                       事件 $A$, $B$ 独立，当且仅当                                       |                                 $$P(A\cap B)=P(A)P(B)$$                                  |                                                               |

## 随机变量

### 常用符号

|                |                                       定义                                        |                                                               形式                                                                |                                 注释                                 |
| -------------- | :-----------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------: |
| **随机变量**       |                             随机变量是将样本空间上的元素映射到实数的函数                              |                                                      $$X: S\to\mathbb{R}$$                                                      |                                                                    |
| **累积分布函数 CDF** | 累计分布函数 $F$ 单调不减且满足 $\lim_{x\to -\infty}F(x)=0$ 和 $\lim_{ n \to +\infty }F(x)=1$ |                                                       $$F(x)=P(X\le x)$$                                                        |                   我们有 $P(a<X\le b) = F(b)-F(a)$                    |
| **概率密度函数 PDF** |                      概率密度函数 $f$ 是 $X$ 取值在两个相邻随机变量的实值之间的概率                       | $$\begin{aligned} &(\text{离散}) \ f(x_{j}) = P(X=x_{j}) \\ &(\text{连续}) \ f(x) = \frac{\mathrm{d} F}{\mathrm{d}x}\end{aligned}$$ | 我们有 $0\le f(x)\le 1$ 且 $\int_{-\infty}^{+\infty}f(x)\mathrm{d}x=1$ |


### 统计量

|            |                                                                                                             形式                                                                                                              |
| ---------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **方差**     |                                                   $$\begin{aligned}\mathrm{Var}(X) &= \mathbb{E}[(X-\mathbb{E}[X])^2]\\ &=\mathbb{E}[X^2]-\mathbb{E}[X]^2\end{aligned}$$                                                    |
| **标准差**    |                                                                                             $$\sigma=\sqrt{ \mathrm{Var}(X) }$$                                                                                             |
| **期望**     |                              $$\begin{aligned} &(\text{离散}) \ \mathbb{E}[X] = \sum_{i} x_{i}f(x_{i}) \\ &(\text{连续}) \ \mathbb{E}[X] = \int_{-\infty}^{+\infty}xf(x)\mathrm{d}x\end{aligned}$$                              |
| **一般期望**   |                        $$\begin{aligned} &(\text{离散}) \ \mathbb{E}[g(X)] = \sum_{i} g(x_{i})f(x_{i}) \\ &(\text{连续}) \ \mathbb{E}[g(X)] = \int_{-\infty}^{+\infty}g(x)f(x)\mathrm{d}x\end{aligned}$$                        |
| **$k$ 阶矩** |                          $$\begin{aligned} &(\text{离散}) \ \mathbb{E}[X^k] = \sum_{i} x_{i}^kf(x_{i}) \\ &(\text{连续}) \ \mathbb{E}[X^k] = \int_{-\infty}^{+\infty}x^kf(x)\mathrm{d}x\end{aligned}$$                          |
| **特征函数**   | $$\begin{aligned} &(\text{离散}) \ \psi(\omega) = \sum_{i} f(x_{i})\mathrm{e}^{\mathrm{i}\omega x_{i}} \\ &(\text{连续}) \ \psi(\omega) = \int_{-\infty}^{+\infty}f(x)\mathrm{e}^{\mathrm{i}\omega x}\mathrm{d}x\end{aligned}$$ |

### 特殊性质

|              |                            定义                             |                                                                                                       形式                                                                                                       |
| ------------ | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **随机变量的变换**  | 设 $X$ 和 $Y$ 被某个函数联系在一起，记 $f_X, f_Y$ 分别为 $X$ 和 $Y$ 的分布函数，有 |                                                                            $$f_{Y}(y)=f_{X}(x)\|\frac{\mathrm{d}x}{\mathrm{d}y}\|$$                                                                            |
| **莱布尼茨积分法则** |     令 $g$ 是 $x$ 和 $c$ 的函数，$a, b$ 是可能依赖于 $c$ 的积分上下限，有      | $$\frac{\partial}{\partial c}\left(\int_{a}^{b} g(x) \mathrm{d}x\right) = \frac{\partial b}{\partial c} g(b) - \frac{\partial a}{\partial c} g(a) + \int_{a}^b \frac{\partial g}{\partial c} (x) \mathrm{d}x$$ |
| **切比雪夫不等式**  |        令 $X$ 为随机变量，期望值为 $\mu$. 对于 $k, \sigma>0$, 有        |                                                                                  $$P(\|X-\mu\|\ge k\sigma)\le \frac{1}{k^2}$$                                                                                  |

## 联合分布随机变量

|            |                                                                  形式                                                                  |
| ---------- | :----------------------------------------------------------------------------------------------------------------------------------: |
| **条件密度**   |                                           $$f_{X\mid Y}(x)=\frac{f_{XY}(x,y)}{f_{Y}(y)}$$                                            |
| **独立性**    |            $$\begin{gathered}\text{随机变量 } X, Y \text{ 独立} \\ \Updownarrow \\ f_{XY}(x,y)=f_{X}(x)f_{Y}(y)\end{gathered}$$            |
| **边缘密度**   |                                     $$f_{X}(x)=\int_{-\infty}^{+\infty}f_{XY}(x,y) \mathrm{d}y$$                                     |
| **联合累计函数** |                      $$F_{{XY}}(x,y)=\int_{-\infty}^{x}\int_{-\infty}^{y}f_{XY}(m,n) \mathrm{d}m \mathrm{d}n$$                       |
| **协方差**    | $$\begin{aligned}\mathrm{Cov}(X,Y) &\triangleq \mathbb{E}[(X-\mu_{X})(Y-\mu_{Y})] \\ &= \mathbb{E}[XY]-\mu_{X}\mu_{Y}\end{aligned}$$ |
| **相关性**    |                         $$\rho_{XY} \triangleq \frac{\mathrm{Cov}(X,Y)}{\sigma_{X}\sigma_{Y}} \in [-1, 1]$$                          |

### 主要分布

| 类型  |                    分布                    |                                                                    PDF                                                                    |    $\mathbb{E}[X]$    |    $\mathrm{Var}(X)$    |
| --- | :--------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: | :-------------------: | :---------------------: |
| 离散  |    $X \sim \mathcal{B}(n,p)$<br>二项分布     |                            $$\begin{aligned}&P(X=x)=\binom{n}{x}p^xq^{n-x}\\ &x \in [\![0,n]\!]\end{aligned}$$                            |        $$np$$         |         $$npq$$         |
| 离散  |    $X \sim \mathrm{Po}(\mu)$<br>泊松分布     |                       $$\begin{aligned}&P(X=x)=\frac{\mu^x}{x!}\mathrm{e}^{-\mu}\\ &x \in \mathbb{N}\end{aligned}$$                       |        $$\mu$$        |         $$\mu$$         |
| 连续  |    $X \sim \mathcal{U}(a,b)$<br>均匀分布     |                                    $$\begin{aligned}&f(x)=\frac{1}{b-a}\\ &x \in [a,b]\end{aligned}$$                                     |   $$\frac{a+b}{2}$$   | $$\frac{(b-a)^2}{12}$$  |
| 连续  | $X \sim \mathcal{N}(\mu,\sigma)$<br>高斯分布 | $$\begin{aligned}&f(x)=\frac{1}{\sqrt{ 2\pi }\sigma} \mathrm{e}^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}\\ &x \in \mathbb{R}\end{aligned}$$ |        $$\mu$$        |      $$\sigma^2$$       |
| 连续  |  $X \sim \mathrm{Exp}(\lambda)$<br>指数分布  |                       $$\begin{aligned}&f(x)=\lambda \mathrm{e}^{-\lambda x}\\ &x \in \mathbb{R}^{+}\end{aligned}$$                       | $$\frac{1}{\lambda}$$ | $$\frac{1}{\lambda^2}$$ |

### 参数估计

#### 随机采样

$n$ 个和 $X$ 独立同分布的随机变量 $X_{1},\dots,X_{n}$ 的集合称为一个随机采样。

#### 估计

根据数据来推断一个统计模型中未知参数值的函数，称为估计。

#### 偏差

估计 $\hat\theta$ 的偏差定义为 $\hat{\theta}$ 分布的期望值和估计对象的真实值之间的差距，即：

$$
\mathrm{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta
$$

*注: 如果有 $\mathbb{E}[\hat{\theta}] = \theta$, 则称这个估计为无偏的。*

#### 中心极限定理

如果随机采样 $X_{1},\dots,X_{n}$ 的均值为 $\mu$, 方差为 $\sigma^2$, 则有：

$$
\bar{X} \underset{ {n\to +\infty} }{ \sim } \mathcal{N}\left( \mu, \frac{\sigma}{\sqrt{ n }} \right)
$$