---
title: 3 流匹配
---
上一节，我们已将流模型和扩散模型构建为以神经网络向量场 $u^\theta_t$ 参数化的生成式模型。然而，我们尚未讨论如何训练它们，即如何优化参数 $\theta$，使生成式模型能输出诸如美观的图像或精彩的视频这些合理的结果。接下来，我们介绍**流匹配**[25, 1, 27]——一种训练 $u^\theta_t$ 的算法，它简洁、可扩展，代表当前的最先进水平。

本节仅考虑流模型：我们有一个神经网络 $u^\theta_t$，通过模拟 ODE 从生成式模型获取样本。

$$
\begin{align*}
X_0 \sim p_{\text{init}},\quad
\mathrm{d}X_t = u^\theta_t(X_t) \mathrm{d}t \qquad \text{（流模型）} \tag{10}
\end{align*}
$$

并以端点 $X_1$ 和 $t=1$ 作为样本。如前所述，我们的目标是使 $X_1$ 服从数据分布 $p_{\text{data}}$，即 $X_1 \sim p_{\text{data}}$。因此，「如何训练」神经网络的问题可归结为：**如何优化 $\theta$，使模拟式 (10) 中的流模型能够生成来自数据分布 $X_{1} \sim p_{\text{data}}$ 的样本**？

## 3.1  条件与边缘概率路径

流匹配的第一步是指定一条**概率路径**。直观上，概率路径定义了从噪声 $p_{\text{init}}$ 到数据 $p_{\text{data}}$ 的渐进插值过程，见图 4。为什么需要这样的路径？回顾可知，我们期望的 ODE 轨迹须满足 $X_0 \sim p_{\text{init}}$（$t=0$）和 $X_1 \sim p_{\text{data}}$（$t=1$）。那么，中间时刻 $0 < t < 1$ 会发生什么？实际上，我们可以自由选择中间过程的行为，这正是概率路径所要数学形式化的核心。

下文对数据点 $z \in \mathbb{R}^d$，用 $\delta_z$ 表示**狄拉克**「分布」。这是最简单的分布：从 $\delta_z$ 采样永远返回 $z$，因而是确定性的。**条件（插值）概率路径**是一族 $\mathbb{R}^d$ 上的分布 $p_t(x|z)$，满足：

$$
\begin{align*}
p_0(\cdot|z) &= p_{\text{init}}, \quad p_1(\cdot|z) = \delta_z \quad \forall z \in \mathbb{R}^d. \tag{11}
\end{align*}
$$

换言之，一条条件概率路径将初始分布 $p_{\text{init}}$ 逐步转化为单一数据点（见图 4）。可将概率路径视为分布空间中的一条轨迹。

每条条件概率路径 $p_t(x|z)$ 都导出一个**边缘概率路径** $p_t(x)$，其定义为：先从数据分布采样 $z \sim p_{\text{data}}$，再从 $p_t(\cdot|z)$ 采样，所得分布即为 $p_t(x)$：

$$
\begin{align*}
z &\sim p_{\text{data}}, \; x \sim p_t(\cdot|z) \;
\Rightarrow x \sim p_t &\blacktriangleright\ \text{从边缘路径采样} \tag{12} \\
p_t(x) &= \int p_t(x|z) p_{\text{data}}(z) dz &\blacktriangleright\ \text{边缘路径的密度} \tag{13}
\end{align*}
$$

注意，我们知道如何从 $p_t$ 采样，但不知道密度值 $p_t(x)$，因为积分难以处理：我们可以计算式 (12)，但无法计算式 (13)。请自行验证：根据式 (11) 中对 $p_t(\cdot|z)$ 的条件，边缘概率路径 $p_t$ 在 $p_{\text{init}}$ 与 $p_{\text{data}}$ 之间插值：

$$
\begin{align*}
p_0 = p_{\text{init}}, \quad p_1 = p_{\text{data}}. \quad&\blacktriangleright\ \text{噪声-数据插值} \tag{14}
\end{align*}
$$

迄今最重要的概率路径示例是高斯概率路径，强烈建议仔细阅读下一示例。

> [!example] 例 3（高斯条件概率路径）
> 
> **高斯概率路径**尤为流行，**大多数最先进模型均采用它**。设 $\alpha_t, \beta_t$ 为**噪声调度器**：两个连续可微的单调函数，满足 $\alpha_0 = \beta_1 = 0$ 且 $\alpha_1 = \beta_0 = 1$。随后定义条件概率路径
> 
> $$
> \begin{equation}p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d) \quad \blacktriangleright\ \text{高斯条件路径} \tag{15}\end{equation}
> $$
> 
> 根据对 $\alpha_t$ 和 $\beta_t$ 所施加的条件，有
> 
> $$
> p_0(\cdot|z) = \mathcal{N}(\alpha_0 z, \beta_0^2 I_d) = \mathcal{N}(0, I_d), \quad p_1(\cdot|z) = \mathcal{N}(\alpha_1 z, \beta_1^2 I_d) = \delta_z,
> $$
> 
> 这里利用了方差为零、均值为 $z$ 的正态分布即为 $\delta_z$ 这一事实。因此，所选 $p_t(x|z)$ 满足式 (11) 对 $p_{\text{init}} = \mathcal{N}(0, I_d)$ 的条件，从而是有效的条件插值路径。图 4 展示了它在图像上的应用。边缘路径 $p_t$ 的采样可表示为：
> 
> $$
> \begin{equation}z \sim p_{\text{data}},\; \epsilon \sim p_{\text{init}} = \mathcal{N}(0, I_d) \;\Rightarrow\; x = \alpha_t z + \beta_t \epsilon \sim p_t\quad \blacktriangleright\ \text{从边缘高斯路径中采样} \tag{16}\end{equation}
> $$
> 
> 直观上，该过程为较小的 $t$ 添加更多噪声，直至 $t = 0$ 时仅剩噪声。图 5 绘制了此类插值路径的示例。

## 3.2 条件与边缘向量场

概率路径 $(p_t)_{0 \le t \le 1}$ 指定了轨迹上的点 $X_t$ *应*服从的分布 $X_t \sim p_t$。此时，这还只是我们「希望」成立的情形。那么，如何找到一个向量场，使它的轨迹 $X_t$ 恰好遵循该概率路径？流匹配显式地构造了这样一个向量场——「边缘向量场」，本节将对此加以说明。

对每个数据点 $z \in \mathbb{R}^d$，令 $u^{\text{target}}_t(\cdot|z)$ 为**条件向量场**。它可以是任何向量场，只要对应的常微分方程能产生条件概率路径 $p_t(\cdot|z)$，即满足
$$
\begin{align*}
X_0 \sim p_{\text{init}},\quad \frac{\mathrm{d}}{\mathrm{d}t}X_t = u^{\text{target}}_t(X_t|z) \;\Rightarrow\; X_t \sim p_t(\cdot|z) \quad (0 \le t \le 1). \tag{17}
\end{align*}
$$
通常，我们仅凭一些代数运算就能解析地求出条件向量场 $u^{\text{target}}_t(\cdot|z)$。以例 3 中的高斯概率路径为例，我们将推导其条件向量场 $u_t(x|z)$ 来说明这一点。

初看之下，条件向量场似乎毫无用处：常微分方程所有轨迹的终点都将坍缩至 $X_1 = z$，等于只是在重新生成已知的数据点 $z$。然而，这个条件向量场正是构建一个真正能从 $p_{\text{data}}$ 生成样本的向量场的基本模块。

> [!theorem] **定理 3（边缘化技巧）**
> 
> 设 $u^{\text{target}}_t(x|z)$ 为式 (17) 所定义的条件向量场，则边缘向量场 $u^{\text{target}}_t(x)$ 定义为
> 
> $$
> u^{\text{target}}_t(x) = \int \frac{u^{\text{target}}_t(x|z)\, p_t(x|z)\, p_{\text{data}}(z)}{p_t(x)} \, dz, \tag{18}
> $$
> 
> 该向量场遵循边缘概率路径，即
> 
> $$
> \begin{equation}X_0 \sim p_{\text{init}},\; \frac{\mathrm{d}}{\mathrm{d}t}X_t = u^{\text{target}}_t(X_t) \;\Rightarrow\; X_t \sim p_t \quad (0 \le t \le 1). \tag{19}\end{equation}
> $$
> 
> 特别地，该常微分方程满足 $X_1 \sim p_{\text{data}}$，因此可以说 $u^{\text{target}}_t$ 将噪声 $p_{\text{init}}$ 转换为数据 $p_{\text{data}}$。

> [!example] **例 3（高斯概率路径的目标 ODE）**
> 
> 如前，设 $p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ 为基于噪声调度器 $\alpha_t, \beta_t$ 的条件概率路径，见式 (15)。令 $\dot{\alpha}_t = \partial_t \alpha_t$，$\dot{\beta}_t = \partial_t \beta_t$ 分别表示 $\alpha_t$ 和 $\beta_t$ 的时间导数。我们将说明，如下定义的条件高斯向量场
> 
> $$
> \begin{equation}u^{\text{target}}_t(x|z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x \tag{20}\end{equation}
> $$
> 
> 构成一个有效的条件向量场。从定理 3 的视角看，若 $X_0 \sim \mathcal{N}(0, I_d)$，则其 ODE 轨迹 $X_t$ 满足 $X_t \sim p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$。图 6 将条件概率路径的真实样本与该流模拟 ODE 轨迹的样本加以比较，直观地确认了这一点：分布互相吻合。下面给出证明。
> 
> **证明.**  首先构建条件流模型 $\psi^{\text{target}}_t(x|z)$：
> 
> $$
> \begin{equation}\psi^{\text{target}}_t(x|z) = \alpha_t z + \beta_t x. \tag{21}\end{equation}
> $$
> 
> 若 $X_t$ 是 $\psi^{\text{target}}_t(\cdot|z)$ 的 ODE 轨迹且 $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I_d)$，则由定义
> 
> $$
> X_t = \psi^{\text{target}}_t(X_0|z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot|z).
> $$
> 
> 可见轨迹的分布与条件概率路径一致，即满足式 (17)。接下来需从 $\psi^{\text{target}}_t(x|z)$ 提取向量场 $u^{\text{target}}_t(x|z)$。根据流的定义（式 (2b)），有
> 
> $$
> \begin{align}\frac{\mathrm{d}}{\mathrm{d}t}\psi^{\text{target}}_t(x|z) &= u^{\text{target}}_t(\psi^{\text{target}}_t(x|z)|z) \quad \forall\, x, z \in \mathbb{R}^d \\ \overset{(i)}\Leftrightarrow\quad \dot{\alpha}_t z + \dot{\beta}_t x &= u^{\text{target}}_t(\alpha_t z + \beta_t x|z) \quad \forall\, x, z \in \mathbb{R}^d \\ \overset{(ii)}\Leftrightarrow\quad \dot{\alpha}_t z + \dot{\beta}_t \left( \frac{x - \alpha_t z}{\beta_t} \right) &= u^{\text{target}}_t(x|z) \quad \forall\, x, z \in \mathbb{R}^d\\ \overset{(iii)}\Leftrightarrow\quad \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x &= u^{\text{target}}_t(x|z) \quad \forall\, x, z \in \mathbb{R}^d\end{align}
> $$
> 
> 步骤 $(i)$ 使用了 $\psi^{\text{target}}_t(x|z)$ 的定义 (21)，步骤 $(ii)$ 做了变量替换 $x \to (x - \alpha_t z)/\beta_t$，步骤 $(iii)$ 则展开了代数运算。最后一个方程正是式 (20) 所定义的条件高斯向量场，结论成立。 $^a$ $\square$
> 
> ---
> $^a$ 也可通过将其代入本节稍后介绍的连续性方程加以复核验证。

定理 3 的图示参见图 6。我们来直观理解边缘向量场。统计学中的贝叶斯规则表明，以下项描述了一个后验分布

$$
\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}=\text{「给定含噪数据 $x$ 下数据点 $z$ 的后验」}
$$

其中 $p_{\text{data}}(z)$ 是先验分布。边缘向量场便简化为一种平均：对每个可能的数据点 $z$，取速度 $u_t(x|z)$（也即引导我们到达 $z$ 的方向），再按我们认为 $x$ 来自 $z$ 的程度加权，最后对所有数据点平均，即得边缘向量场。

本节余下部分将严格化这一直觉，并证明定理 3。主要的数学工具是连续性方程，一个数学和物理学中的基本方程。定义散度算子 $\operatorname{div}$ 为

$$
\operatorname{div}(v_t)(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v_t^i(x) \tag{22}
$$

其中 $v_t^i$ 是 $v_t$ 的第 $i$ 个坐标。

> [!theorem] **定理 4 (连续性方程)**  
> 考虑向量场 $u_\text{target}^t$ 的流模型，其中 $X_0 ∼ p_\text{init} = p_0$。那么对所有 $0 \le t \le 1$，当且仅当 $X_t ∼ p_t$ 时，  
> $$
> \begin{align*}
> \partial_t p_t(x) = -\operatorname{div}(p_t u_t^{\text{target}})(x),\quad \forall x \in \mathbb{R}^d,\ 0 \le t \le 1, \tag{23}
> \end{align*}
> $$
> 这里 $\partial_t p_t(x) = \frac{\mathrm{d}}{\mathrm{d}t} p_t(x)$ 表示 $p_t(x)$ 的时间导数。上式即连续性方程，也称作方程 (23)。

有数学背景的读者，我们在 Section B 中给出了连续性方程的自包含证明。继续之前，先直观上理解它。左侧 $∂_t p_t(x)$ 描述 $x$ 处概率 $p_t(x)$ 随时间的变化，而这种变化对应概率质量的净流入。流模型中，粒子 $X_t$ 沿向量场 $u_t^{\text{target}}$ 运动。物理学上，散度衡量向量场的净流出量，其相反数即净流入量。将该量乘以 $x$ 处的总概率质量，得到 $−\operatorname{div}(p_t u_t)$，它衡量概率质量的总流入量。概率质量守恒（总积分恒为 1），故方程左右两侧自然相等！下面证明定理 3 的边缘化技巧。

**定理 3 的证明。** 由定理 4，只需证明方程 (18) 定义的边缘向量场 $u_t^{\text{target}}$ 满足连续性方程。直接计算如下：

$$
\begin{align*}
\partial_t p_t(x) \overset{(i)}{=} \partial_t \int p_t(x|z) p_{\text{data}}(z) \, dz &= \int \partial_t p_t(x|z) p_{\text{data}}(z) \, dz \\
&\overset{(ii)}{=} \int -\operatorname{div}\left(p_t(\cdot|z) u_t^{\text{target}}(\cdot|z)\right)(x) \, p_{\text{data}}(z) \, dz \\
&\overset{(iii)}{=} -\operatorname{div}\left( \int p_t(x|z) u_t^{\text{target}}(x|z) p_{\text{data}}(z) \, dz \right) \\
&\overset{(iv)}{=} -\operatorname{div}\left( p_t(x) \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} \, dz \right)(x) \\
&\overset{(v)}{=} -\operatorname{div}\left( p_t u_t^{\text{target}} \right)(x),
\end{align*}
$$

其中 ($i$) 使用了方程 (12) 中 $p_t(x)$ 的定义；($ii$) 使用了条件概率路径的连续性方程 $p_t(·|z)$；($iii$) 通过方程 (22) 交换积分与散度算子；($iv$) 乘除 $p_t(x)$；($v$) 使用了方程 (18)。上述推导表明连续性方程对 $u_t^{\text{target}}$ 成立。由定理 4，方程 (19) 得证。 $\square$

