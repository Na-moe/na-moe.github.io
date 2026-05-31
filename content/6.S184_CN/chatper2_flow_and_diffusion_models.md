---
title: 2 流与扩散模型
---
上一节将生成式建模形式化为从数据分布 $p_\text{data}$ 中采样，进而把目标定为：构建一个生成式模型，即一种能返回样本 $z \sim p_\text{data}$ 的算法。本节将描述如何通过模拟适当构造的微分方程来构建生成式模型。例如，流匹配与扩散模型分别涉及模拟**常微分方程**（ODEs）和**随机微分方程**（SDEs）。因此，本节旨在定义并构造这些生成式模型，它们将在后续笔记中反复用到。具体而言，本节首先定义 ODEs 和 SDEs，并讨论其模拟方法；其次描述如何用深度神经网络参数化 ODEs 与 SDEs。由此将引出流模型和扩散模型的定义，以及从中采样的基本算法。后续章节再探讨如何训练这些模型。

## 2.1 流模型

先定义**常微分方程（ODEs）**。一个 ODE 的解由一条**轨迹**定义，其形式为

$$
X: [0,1] \to \mathbb{R}^d,\quad t \mapsto X_t,
$$

它将时间 $t$ 映射到空间 $\mathbb{R}^d$ 中的某一位置。每个常微分方程由一个**向量场** $u$ 定义，即形如下式的函数

$$
u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d,\quad (x,t) \mapsto u_t(x),
$$

对每个时刻 $t$ 和位置 $x$，它给出一个向量 $u_t(x) \in \mathbb{R}^d$，指定空间中的速度（见图 1）。

常微分方程为轨迹设定了一个约束：希望找到一条轨迹 $X$，它沿着向量场 $u_t$ 的「方向线」运动，并从点 $x_0$ 出发。这条轨迹可形式化为以下方程的解：

^eq1
$$
\begin{align*}
\frac{\mathrm{d}}{\mathrm{d}t}X_t &= u_t(X_t)  &\blacktriangleright\ \text{ODE}  \tag{1a}\\
X_0 &= x_0  &\blacktriangleright\ \text{初始条件}  \tag{1b}
\end{align*}
$$



式 [[chatper2_flow_and_diffusion_models#^eq1|(1a)]] 要求 $X_t$ 的导数由 $u_t$ 给定的方向决定。式 [[chatper2_flow_and_diffusion_models#^eq1|(1b)]] 要求轨迹在时刻 $t=0$ 从 $x_0$ 出发。

自然会问：若 $t=0$ 时从 $X_0 = x_0$ 出发，在时刻 $t$ 会到达何处（即 $X_t$ 是什么）？这个问题的答案由一个称为**流**（flow）的函数给出，它就是该 ODE 的解：

^eq2
$$
\begin{align*}
\psi: \mathbb{R}^d \times [0,1] &\to \mathbb{R}^d,\quad (x_0,t) \mapsto \psi_t(x_0) \tag{2a} \\
\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0) &= u_t(\psi_t(x_0)) &\blacktriangleright \text{流的 ODE} \tag{2b}\\
\psi_0(x_0) &= x_0 &\blacktriangleright\ \text{流的初始条件} \tag{2c}
\end{align*}
$$

给定初始条件 $X_0 = x_0$，ODE 的轨迹可由 $X_t = \psi_t(X_0)$ 得出。因此，向量场、ODE 和流直观上是同一对象的三种描述：**向量场定义了 ODE，其解即为流**。与所有方程类似，应当对 ODE 提出如下问题：解是否存在？若存在，是否唯一？数学中有一个基本结论：只要对 $u_t$ 施加较弱的假设，这两个问题的答案都是「是」。

> [!theorem] **定理 1（流的存在性与唯一性）**  ^theorem1
> 若 $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ 有界可导且连续可微，则式 [[chatper2_flow_and_diffusion_models#^eq2|(2)]] 中的常微分方程有唯一解，由流 $\psi_t$ 给出。此时对所有 $t$，$\psi_t$ 是**微分同胚**，即 $\psi_t$ 连续可微且其逆映射 $\psi_t^{-1}$ 也连续可微。

注意，机器学习中，通常用神经网络参数化 $u_t(x)$，它们总具有有界导数，因此流存在且唯一所需的假设几乎总能满足。[[chatper2_flow_and_diffusion_models#^theorem1|定理 1]] 非但无需担忧，反而是个好消息：**在所关注的场景中，流始终存在，且为 ODE 的唯一解**。证明可参见。

^eq3
> [!example] **例 1（线性向量场）**  
> 考虑一个简单的向量场 $u_t(x)$，它是 $x$ 的线性函数：对 $\theta > 0$，$u_t(x) = -\theta x$。函数
> 
> $$\begin{equation}
> \psi_t(x_0) = \exp(-\theta t) x_0 \tag{3}
> \end{equation}$$
> 
> 定义了一个流 $\psi$，它是式 [[chatper2_flow_and_diffusion_models#^eq2|(2)]] 中常微分方程的解。验证如下：$\psi_0(x_0) = x_0$，并且
> 
> $$ \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0)&\overset{(3)}{=} \frac{\mathrm{d}}{\mathrm{d}t} \big( \exp(-\theta t) x_0 \big) \\&\overset{(i)}{=} -\theta \exp(-\theta t) x_0 \\&\overset{(3)}{=} -\theta \psi_t(x_0) = u_t(\psi_t(x_0)),
> \end{aligned}$$
> 
> 其中步骤 $(i)$ 应用了链式法则。图 3 直观展示了该流以指数速度收敛到 $0$。

### 模拟 ODE

通常，若 $u_t$ 不像前例一样简单，便无法显式计算流 $\psi_t$。此时需借助数值方法模拟常微分方程。所幸这是数值分析中一个经典而成熟的课题，已有大量强大的方法。其中最简单且最直观的是 **Euler 法**：初始化 $X_0 = x_0$，然后按以下方式更新：
$$
X_{t+h} = X_t + h  u_t(X_t) \qquad (t = 0, h, 2h, 3h, \dots, 1-h) \tag{4}
$$

其中步长 $h = n^{-1} > 0$，$n \in \mathbb{N}$ 为仿真步数。对这类问题，Euler 法已足够适用。为了展示更复杂的方法，考虑 **Heun 法**，其更新规则定义如下：

$$
\begin{align*}
X_{t+h}' &= X_t + h  u_t(X_t) &\blacktriangleright\ \text{新状态的初始猜测（同 Euler 法）}\\
X_{t+h} &= X_t + \frac{h}{2} \big( u_t(X_t) + u_{t+h}(X_{t+h}') \big) &\blacktriangleright\ \text{用当前与猜测状态的平均更新}
\end{align*}
$$

直观上看，Heun 法先初步估计下一步的可能值 $X_{t+h}'$，再用更新后的估计来修正初始选取的方向。

### 流模型

现在可以通过把向量场构建为**神经网络向量场** $u_t^\theta$，来构造生成式模型。目前，$u_t^\theta$ 仅表示一个参数化函数 $u_t^\theta : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$，参数为 $\theta$。稍后将讨论具体的神经网络结构选择。需牢记目标是从分布 $p_\text{data}$ 中采样 $z \sim p_\text{data}$。特别地，这些样本必须是随机的。但请注意，常微分方程本身完全确定，并不随机。为引入随机性，只需让初始条件 $X_0$ 随机。具体来说，需选择一个**初始分布** $p_\text{init}$，多数情况下设 $p_\text{init} = \mathcal{N}(0, I_d)$，即简单的标准高斯分布。关键之处在于，无论选择何种分布，在推理时必须易于采样。**流模型**由此常微分方程描述：

$$
\begin{align*}
X_0 &\sim p_\text{init} &\blacktriangleright\ \text{随机初始化}\\
\frac{\mathrm{d}}{\mathrm{d}t} X_t &= u_t^\theta(X_t) &\blacktriangleright\ \text{ODE}
\end{align*}
$$

目标是让轨迹终点 $X_1$ 服从分布 $p_\text{data}$，即 

$$
X_1 \sim p_\text{data}\ \Leftrightarrow\ \psi_1^\theta(X_0) \sim p_\text{data}
$$

其中 $\psi_t^\theta$ 是由 $u_t^\theta$ 确定的流。注意，虽然名为*流模型*，**神经网络参数化的是向量场，而非流本身**。计算流需要模拟常微分方程。[[chatper2_flow_and_diffusion_models#^algo1|算法 1]] 概括了从流模型中采样的流程。

^algo1
<div style="border-top: 2px solid; border-bottom: 1px solid;"> <b>算法 1</b> 使用 Euler 法从流模型中采样</div>  

**Require:** 神经网络向量场 $u_{t}^\theta$，步数 $n$  
 1: 设置 $t = 0$  
 2: 设置步长 $h=\frac{1}{n}$  
 3: 采样一个 $X_{0} \sim p_{\text{init}}$  
 4: **for** $i=1, \dots, n$ **do**  
 5:$\quad$ $X_{t+h}=X_{t}+hu_{t}^\theta\!\left( X_{t} \right)$  
 6:$\quad$ 更新 $t\leftarrow t+h$  
 7: **end for**  
 8: **return** $X_{1}$  
 
<hr style="
    border: 0;
    border-top: 1px solid;
">

## 2.2 扩散模型

随机微分方程通过**随机**轨迹，将常微分方程的确定性轨迹加以扩展。随机轨迹通常称为**随机过程** $(X_t)_{0 \le t \le 1}$，由以下形式给出：

$$
\begin{align*}
\text{对每个 } 0 \le t \le 1,\ X_{t} \text{ 都是随机变量}& \\
X: [0,1] \to \mathbb{R}^d,\ \text{对 } X  \text{ 的每条抽样},\ t \mapsto X_t \text{ 都构成一条随机轨迹}&
\end{align*}
$$

特别地，对同一随机过程模拟两次，因动态机制内具随机性，可能得到不同结果。

### 布朗运动

SDE 借由**布朗运动**构建而成。布朗运动是源自物理扩散过程研究的一个基本随机过程，可视为一种连续的随机游走。

定义如下：布朗运动 $W = (W_t)_{0 \le t \le 1}$ 是一个随机过程，满足 $W_0 = 0$，轨道 $t \mapsto W_t$ 连续，并满足以下两个条件：

1. **正态增量**：对所有 $0 \le s < t$，$W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$，即增量服从高斯分布，方差随时间线性增加（$I_d$ 为单位矩阵）。
2. **独立增量**：对任意 $0 \le t_0 < t_1 < \dots < t_n = 1$，增量 $W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}}$ 相互独立。

布朗运动也称**维纳过程**，故用「$W$」表示。[^1] 可通过设定步长 $h > 0$，令 $W_0 = 0$ 并按下式更新，轻松地近似模拟布朗运动：

$$
W_{t+h} = W_t + \sqrt{h}\epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, I_d) \qquad (t = 0, h, 2h, \dots, 1-h) \tag{5}
$$

图 2 展示了几条布朗运动的示例轨迹。布朗运动之于随机过程，犹如高斯分布之于概率分布，居于核心地位。从金融、统计物理到流行病学，布朗运动的研究在机器学习之外有着广泛的应用。例如，金融领域用布朗运动模拟复杂金融工具的价格。仅作为数学构造，布朗运动同样引人入胜：尽管其路径连续，可以一笔画成，却具有无限长度，永远无法画完。

### 从 ODEs 到 SDEs

SDE 的思想，是在 ODE 确定性动力学的基础上添加布朗运动驱动的随机动力学。既然一切皆随机，便不能再像式 [[chatper2_flow_and_diffusion_models#^eq1|(1a)]] 那样取导数，需要找到 ODE **不使用导数的等价表述**。为此，将 ODE 轨迹 $(X_t)_{0\le t\le 1}$ 改写如下。由导数定义，

$$
\begin{align*}

\frac{\mathrm{d}}{\mathrm{d}t}X_t &= u_t(X_t) &\blacktriangleright\ \text{通过导数表示}\\
\overset{(i)}{\Leftrightarrow}\; \frac{1}{h}(X_{t+h} - X_t) &= u_t(X_t) + R_t(h)\\
\Leftrightarrow\;X_{t+h} &= X_t + hu_t(X_t) + hR_t(h) &\blacktriangleright\ \text{通过无穷小量更新表示}
\end{align*}
$$

其中 $R_t(h)$ 是对小量 $h$ 可忽略的函数，即 $\lim_{h\to 0} R_t(h) = 0$，步骤 $(i)$ 应用了导数的定义。

上述推导重申了一个已知事实：ODE 轨迹 $(X_t)_{0\le t\le 1}$ 在每个时间步长沿方向 $u_t(X_t)$ 迈出一小步。现在修改最后一个等式以引入随机性：SDE 轨迹 $(X_t)_{0\le t\le 1}$ 在每一步既沿 $u_t(X_t)$ 移动，又叠加来自布朗运动的随机扰动：

^eq6
$$
X_{t+h} = X_t + \underbrace{ hu_t(X_t) }_{ \text{确定性} } + \underbrace{ \sigma_t (W_{t+h} - W_t) }_{ \text{随机性} } + \underbrace{ hR_t(h) }_{ \text{误差项} } \tag{6}
$$

其中 $\sigma_t \ge 0$ 为**扩散系数**，$R_t(h)$ 为随机误差项，满足 $\mathbb{E}\big[\|R_t(h)\|^2\big]^{1/2} \to 0$（$h \to 0$）。上式定义了一个**随机微分方程**，通常记为

^eq7
$$
\begin{align*}
\mathrm{d}X_t &= u_t(X_t)\mathrm{d}t + \sigma_t\mathrm{d}W_t \tag{7a}\\
X_0 &= x_0 \tag{7b}
\end{align*}
$$

务必牢记，这里的「$\mathrm{d}X_t$」符号只是式 (6) 的非正式记法。遗憾的是，SDE 不再拥有流映射 $\phi_t$，因为演化本身是随机的，$X_t$ 不再由 $X_0 \sim p_\text{init}$ 完全决定。尽管如此，与 ODE 类似，有

> [!theorem] **定理 2（SDE 解的存在性与唯一性）**  
> 若 $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ 连续可微且导数有界，$\sigma_t$ 连续，则式 [[chatper2_flow_and_diffusion_models#^eq7|(7)]] 中的 SDE 存在唯一解，由满足式 [[chatper2_flow_and_diffusion_models#^eq6|(6)]] 的随机过程 $(X_t)_{0\le t\le 1}$ 给出。

随机微积分课程将花数节课严格证明此定理，并以完全数学严谨的方式构建 SDE：从基本原理出发构造布朗运动，再通过随机积分构造过程 $X_t$。鉴于本课程侧重机器学习，更技术性的细节可参考文献 [29]。最后，每个 ODE 也都是 SDE，只需令扩散系数 $\sigma_t = 0$。因此，后文**讨论 SDE 时，均将 ODE 视为其特例**。

> [!example] **例 2（Ornstein–Uhlenbeck 过程）**  
> 考虑常数扩散系数 $\sigma_t = \sigma \ge 0$ 和常数线性漂移项 $u_t(x) = -\theta x$，对于 $\theta > 0$，得到如下 SDE：
> 
> $$\begin{equation}
> \mathrm{d}X_t = -\theta X_t \mathrm{d}t + \sigma \mathrm{d}W_t. \tag{8} 
> \end{equation}$$
> 
> 上述 SDE 的解 $(X_t)_{0\le t\le 1}$ 称为 **Ornstein–Uhlenbeck 过程**，简称 OU 过程。图 3 对其进行了可视化。向量场 $-\theta x$ 将过程推回中心 $0$，因为漂移项始终指向与当前位置相反的方向；扩散系数 $\sigma$ 则持续注入更多噪声。若模拟该过程直至 $t \to \infty$，它将收敛到高斯分布 $\mathcal{N}\!\left(0, \sigma^2/(2\theta)\right)$。注意， $\sigma = 0$ 时，得到一个线性向量场所定义的流，该流已在式 [[chatper2_flow_and_diffusion_models#^eq3|(3)]] 中讨论过。

### 模拟 SDE

若仍对 SDE 的抽象定义感到困惑，不必担心。通过回答下面这个问题，可以获得更直观的理解：如何模拟一个 SDE？最简单的模拟方法称为 **Euler–Maruyama 法**，它对 SDE 的作用正如 Euler 法对 ODE 的作用。采用 Euler–Maruyama 法，初始化 $X_0 = x_0$，然后迭代更新：

$$
\begin{align*}
X_{t+h} &= X_t + h  u_t(X_t) + \sqrt{h}  \sigma_t  \epsilon_t, \tag{9}\\
\epsilon_t &\sim \mathcal{N}(0, I_d),
\end{align*}
$$

其中步长 $h = n^{-1} > 0$，$n \in \mathbb{N}$。换言之，用 Euler–Maruyama 法模拟时，每次沿 $u_t(X_t)$ 方向迈一小步，并加上些许由 $\sqrt{h}\sigma_t$ 缩放的高斯噪声。随附实验等此类 SDE 模拟中，通常都使用 Euler–Maruyama 法。

### 扩散模型

现在，可以像构建 ODE 模型那样，借由 SDE 构造生成式模型。请记住，目标是将简单分布 $p_\text{init}$ 转换为复杂分布 $p_\text{data}$。与 ODE 类似，以 $X_0 \sim p_\text{init}$ 为初始值对 SDE 进行随机模拟，是实现这一变换的自然选择。为参数化此 SDE，只需通过神经网络 $u_t^\theta$ 对其核心成分，向量场 $u_t$，进行参数化。于是，**扩散模型**由以下形式给出：

$$
\begin{align*}
X_0 &\sim p_\text{init} &\blacktriangleright\ \text{随机初始化}\\
\mathrm{d}X_t &= u_t^\theta(X_t)\mathrm{d}t + \sigma_t\mathrm{d}W_t &\blacktriangleright\ \text{SDE}
\end{align*}
$$

[[chatper2_flow_and_diffusion_models#^algo2|算法 2]] 描述了如何通过 Euler–Maruyama 法从扩散模型采样。

^algo2
<div style="border-top: 2px solid; border-bottom: 1px solid;"> <b>算法 2</b> 使用 Euler–Maruyama 法从扩散模型中采样</div>  

**Require:** 神经网络向量场 $u_{t}^\theta$，步数 $n$，扩散系数 $\sigma_{t}$  
 1: 设置 $t = 0$  
 2: 设置步长 $h=\frac{1}{n}$  
 3: 采样一个 $X_{0} \sim p_{\text{init}}$  
 4: **for** $i=1, \dots, n$ **do**  
 5:$\quad$ 采样一个 $\epsilon \sim \mathcal{N}(0,I_{d})$  
 6:$\quad$ $X_{t+h}=X_{t}+hu_{t}^\theta\!\left( X_{t} \right)+\sigma_{t}\sqrt{ h }\epsilon$  
 7:$\quad$ 更新 $t\leftarrow t+h$  
 8: **end for**  
 9: **return** $X_{1}$  
 
<hr style="
    border: 0;
    border-top: 1px solid;
">

现将本节结果总结如下。

> [!summary] **总结 2（SDE 生成式模型）**
> 
> 在本文中，**扩散模型**由参数为 $\theta$ 的神经网络 $u_t^\theta$ 和固定的扩散系数 $\sigma_t$ 组成，该神经网络用于参数化向量场：
> 
> $$\begin{aligned}
> \text{神经网络：} &\quad u^\theta : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, \quad (x,t) \mapsto u_t^\theta(x), \quad \text{参数为 } \theta \\\text{固定项：} &\quad \sigma_t : [0,1] \to [0,\infty), \quad t \mapsto \sigma_t
> \end{aligned}$$
> 
> 从 SDE 模型获取样本（即生成对象）的流程如下：
> 
> $$\begin{aligned}
> \text{初始化：} &\quad X_0 \sim p_\text{init} &\blacktriangleright \text{从高斯分布这类简单分布初始化}\\\text{模拟：} &\quad \mathrm{d}X_t = u_t^\theta(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t &\blacktriangleright \text{从 0 到 1 模拟 SDE}\\\text{目标：} &\quad X_1 \sim p_\text{data} &\blacktriangleright \text{使 } X_1 \text{ 服从分布 } p_\text{data}
> \end{aligned}$$
> 
> $\sigma_t = 0$ 的扩散模型即为**流模型**。

[^1]: 诺伯特 · 维纳是麻省理工学院任教的著名数学家，至今在麻省理工学院数学系仍可见其肖像悬挂。
