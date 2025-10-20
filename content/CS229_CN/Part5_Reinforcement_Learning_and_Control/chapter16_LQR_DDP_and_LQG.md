---
title: 第 16 章 LQR, DDP 与 LQG
---
| [[chapter15_reinforcement_learning\|上一章]] | [[CS229_CN/index#目录\|目录]] | [[chapter17_policy_gradient_REINFORCE\|下一章]] |
| :---------------------------------------: | :-----------------------: | :------------------------------------------: |

## 16.1 有限时间范围的 MDP

第 [[chapter15_reinforcement_learning|15]] 章中定义了马尔可夫决策过程，并讨论了简化设置下的价值迭代/策略迭代。具体而言，引入了定义最优策略 $\pi^*$ 的最优价值函数 $V^*$ 的 **最优贝尔曼方程 (optimal Bellman equation)**。

$$
V^*(s) = R(s) + \max_{a \in A}\gamma \sum_{s' \in S} P_{sa}(s')V^*(s')
$$

回想一下，从最优价值函数中能恢复最优策略$\pi^*$, 如下所示：

$$
\pi^*(s) = \operatorname*{argmax}_{a \in A} \sum_{s' \in S} P_{sa}(s')V^*(s')
$$

在本章中，将采用更一般的设置：

1. 为了同时适用于离散和连续的情况。因此，将使用

$$
\begin{align*}
    &\mathbb{E}_{s' \sim P_{sa}}[V^{\pi^*}(s')] \quad \text{而非} \\
    &\sum_{s' \in S} P_{sa}(s')V^{\pi^*}
\end{align*}
$$

* 这意味着将取下一状态价值函数的期望。在离散情况下，可以将期望改写为对所有状态的求和。在连续情况下，可以将期望改写为对所有状态的积分。符号 $s' \sim P_{sa}$ 表示 $s'$ 是从分布 $P_{sa}$ 中采样的。

2. 假设奖励同时取决于 **状态和动作 (both states and actions)**。换句话说，$R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$. 这意味着计算最优动作的先前机制变为

$$
\pi^*(s) = \underset{a \in \mathcal{A}}{\text{argmax}} \ R(s, a) + \gamma \mathbb{E}_{s' \sim P_{sa}} [V^{\pi^*}(s')]
$$

3. 不考虑无限时间范围的 MDP，而是假设有一个 **有限时间范围的 MDP (finite horizon MDP)**，定义为一个元组

$$
(\mathcal{S}, \mathcal{A}, P_{sa}, T, R)
$$

* 其中 $T > 0$ 是 **时间范围 (time horizon)** (例如 $T=100$)。在这种设置下，对回报的定义将略有不同：

$$
R(s_0, a_0) + R(s_1, a_1) + \cdots + R(s_T, a_T)
$$

* 而不是 (无限时间范围情况下的)

$$
\begin{align*}
    R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \cdots = \sum_{t=0}^{\infty} R(s_t, a_t)\gamma^t
\end{align*}
$$

* *折现因子 $\gamma$ 呢*？请记住，引入 $\gamma$  (部分原因) 是为了确保无限项求和是有限且良定义的。如果奖励以常数 $\bar{R}$ 为界，则回报是以下式为界的

$$
\left| \sum_{t=0}^{\infty} R(s_t)\gamma^t \right| \leq \bar{R} \sum_{t=0}^{\infty} \gamma^t
$$

* 发现这里是一个几何级数！在这里，由于回报是有限项求和，折现因子 $\gamma$ 不再是必需的。
* 在这种新设置中，情况表现得相当不同。首先，最优策略 $\pi^*$ 可能是非平稳的，这意味着 **它随时间变化 (it changes over time)**。换句话说，现在有

$$
\pi^{(t)}: \mathcal{S} \to \mathcal{A}
$$

* 其中上标 $(t)$ 表示时间步 $t$ 的策略。遵循策略 $\pi^{(t)}$ 的有限时间范围 MDP 的动态过程如下：从某个状态 $s_0$ 开始，根据时间步 0 的策略 $\pi^{(0)}(s_0)$ 采取某个动作 $a_0 := \pi^{(0)}(s_0)$. MDP 根据 $P_{s_0 a_0}$ 转换到后继状态 $s_1$. 然后根据时间步 1 的新策略 $\pi^{(1)}(s_1)$ 选择另一个动作 $a_1 := \pi^{(1)}(s_1)$，依此类推……
* *为什么最优策略在有限时间范围设置下会是非平稳的*？直观地说，由于有有限数量的动作要执行，实际可能希望根据所处的环境位置以及剩余的时间来采取不同的策略。想象一个有 2 个目标的网格，奖励分别为 $+1$ 和 $+10$. 一开始可能希望采取行动以得到 $+10$ . 但如果经过一些步骤，动态过程以某种方式接近了 $+1$, 并且没有足够的剩余步数来达到 $+10$, 那么更好的策略将是瞄准 $+1$......
4. 这一现象导致了 **时间依赖的动态过程 (time dependent dynamics)**

$$
s_{t+1} \sim P^{(t)}_{s_t, a_t}
$$

* 这意味着转移分布 $P^{(t)}_{s_t, a_t}$ 随时间变化。关于 $R^{(t)}$ 也是一样。请注意，这种设置更好地模拟了现实生活。例如在汽车驾驶时，油箱会变空，交通状况会变化等等。综上，对有限时间范围 MDP 使用以下一般表述

$$
(\mathcal{S}, \mathcal{A}, P^{(t)}_{sa}, T, R^{(t)})
$$

* **备注.**  注意上式等价于将时间加入到状态中。
* 在时间 $t$ 处，策略 $\pi$ 的价值函数定义方式与之前相同，即从状态 $s$ 开始，遵循策略 $\pi$ 生成的轨迹的期望：

$$
V_t(s) = \mathbb{E}[R^{(t)}(s_t, a_t) + \cdots + R^{(T)}(s_T, a_T) | s_t = s, \pi]
$$

现在，问题是

$$
\begin{gathered}
\textit{如何在有限时间范围设置下找到最优价值函数} \\
V_t^*(s) = \max_{\pi} V_t^{\pi}(s)
\end{gathered}
$$
事实上，贝尔曼方程的价值迭代是为了 **动态规划 (Dynamic Programming)** 而设计的。这不足为奇，因为贝尔曼是动态规划的创始人之一，而贝尔曼方程与该领域密切相关。为了理解如何通过采用基于迭代的方法来简化问题，有如下观察：

1. 注意，在博弈结束时 (对于时间步 $T$)，最优值是显而易见的

^eq16-1
$$
\forall s \in \mathcal{S}: \ V_T^*(s) := \max_{a \in \mathcal{A}} R^{(T)}(s, a) \tag{16.1}
$$

2. 对于另一个时间步 $0 \leq t < T$, 如果假设已知下一个时间步的最优价值函数 $V_{t+1}^*$, 那么就有

^eq16-2
$$
\forall t < T, s \in \mathcal{S}: \ V_t^*(s) := \max_{a \in \mathcal{A}} \left[ R^{(t)}(s, a) + \mathbb{E}_{s' \sim P_{sa}^{(t)}} [V_{t+1}^*(s')] \right] \tag{16.2}
$$

考虑上述结果，可以提出一个巧妙的算法来解决最优价值函数：

1. 使用公式 [[chapter16_LQR_DDP_and_LQG#^eq16-1|(16.1)]] 计算 $V_T^*$.
2. 对于 $t = T-1, \dots, 0$:
	* 使用 $V_{t+1}^*$ 和公式 [[chapter16_LQR_DDP_and_LQG#^eq16-2|(16.2)]] 计算 $V_t^*$.

**旁注.**  可以将标准价值迭代视作这种一般情况的特例 (即不将时间视为状态的一部分)。结果表明，在标准设置中，如果运行价值迭代 $T$ 步，将得到最优价值迭代的 $\gamma^T$ 近似 (几何收敛)。下面结果的证明请参见习题集 4：

<u>定理.</u>  设 $B$ 是贝尔曼更新算子，然后记 $\|f(x)\|_\infty := \sup_x |f(x)|$. 令 $V_t$ 表示第 $t$ 步的价值函数。则

$$
\begin{align*}
    \|V_{t+1} - V^*\|_\infty 
      &= \|B(V_{t}) - V^*\|_\infty\\
      &\le \gamma \|V_{t} - V^*\|_\infty\\
      &\le \gamma^t \|V_1 - V^*\|_\infty
\end{align*}
$$

所以贝尔曼更新算子 $B$ 是一个 $\gamma$-收缩算子。

## 16.2 线性二次调节器 (LQR)

本节将讨论第 [[chapter16_LQR_DDP_and_LQG#16.1 有限时间范围的 MDP|16.1]] 节所述的有限时间范围设定中的一个特例，其 **精确解 (exact solution)** 是可 (容易) 处理的。该模型在机器人中被广泛使用。

首先，描述该模型的假设。考虑状态和动作是连续的情况，其中

$$
\mathcal{S} = \mathbb{R}^d, \mathcal{A} = \mathbb{R}^d
$$

并且假设 **线性转移 (linear transitions)** (带噪声)

$$
s_{t+1} = A_t s_t + B_t a_t + w_t
$$

其中 $A_t \in \mathbb{R}^{d \times d}$, $B_t \in \mathbb{R}^{d \times d}$ 是矩阵，$w_t \sim \mathcal{N}(0, \Sigma_t)$ 是某种高斯噪声 (均值为**零**)。正如后文所示，只要噪声具有零均值，它就不会影响最优策略！

还假设模型具有 **二次奖励 (quadratic rewards)**

$$
R^{(t)}(s_t, a_t) = -s_t^\top U_t s_t - a_t^\top W_t a_t
$$

其中 $U_t \in \mathbb{R}^{d \times n}$, $W_t \in \mathbb{R}^{d \times d}$ 是正定矩阵 (意味着奖励总是**负**的)。

**备注.**  请注意，奖励的二次性等价于希望状态接近原点 (以获得较高奖励)。例如，如果 $U_t = I_d$ (单位矩阵) 和 $W_t = I_d$，则 $R_t = -\|s_t\|^2 - \|a_t\|^2$, 这意味着希望采取平滑的动作 ($a_t$ 的范数小) 以回到原点 ($s_t$ 的范数小)。这可以模拟一辆汽车试图保持在车道中间而不进行冲动性移动……

定义了 LQR 模型的假设之后，接下来介绍 LQR 算法的 2 个步骤。

**步骤 1** 假设不知道矩阵 $A, B, \Sigma$. 为了估计它们，可以借鉴强化学习一章中价值逼近部分的思想。首先，从任意策略中收集转移。然后，使用线性回归来找到 $\arg\min_{A,B} \sum_{i=1}^n \sum_{t=0}^{T-1} \left|s_{t+1}^{(i)} - \left(A s_t^{(i)} + B a_t^{(i)}\right)\right|^2$. 最后，使用高斯判别分析一节中学到的技术来学习 $\Sigma$.

**步骤 2** 假设模型参数已知 (给定或通过步骤 1 估计)，可以使用动态规划推导出最优策略。

$\ \quad$ 换句话说，给定

$$
\begin{cases}
        s_{t+1} &= A_t s_t + B_t a_t + w_t \quad A_t, B_t, U_t, W_t, \Sigma_t \ \text{已知}\\
        R^{(t)}(s_t, a_t) &= -s_t^\top U_t s_t - a_t^\top W_t a_t
    \end{cases}
$$

$\ \quad$ 计算 $V_t^*$。根据第 [[chapter16_LQR_DDP_and_LQG#16.1 有限时间范围的 MDP|16.1]] 节，可以应用动态规划，得到

$\ \quad$ 1. **初始化**

$\ \quad\ \quad$ 对于最后一个时间步 $T$，

$$
\begin{align*}
    V_T^*(s_T)
        &= \max_{a_T \in \mathcal{A}} R_T(s_T, a_T) \\
        &= \max_{a_T \in \mathcal{A}} -s_T^\top U_T s_T - a_T^\top W_T a_T\\
        &= -s_T^\top U_T s_T \quad \text{(最大化时 $a_T=0$)}
\end{align*}
$$

$\ \quad$ 2. **迭代步骤**

$\ \quad\ \quad$ 令 $t < T$. 假设已知 $V_{t+1}^*$.

$\ \quad\ \quad$ <u>事实 1:</u> 可以证明，如果 $V_{t+1}^*$ 是 $s_t$ 的二次函数，则 $V_t^*$ 也是二次函数。

$\ \quad\ \quad\ \quad$ 换句话说，存在某个矩阵 $\Phi$ 和某个标量 $\Psi$ 使得下式成立

$$
\begin{align*}
    &\text{如果}\  V_{t+1}^*(s_{t+1}) = s_{t+1}^\top \Phi_{t+1} s_{t+1} + \Psi_{t+1} \\
    &\text{则}\  V_t^*(s_t) = s_t^\top \Phi_t s_t + \Psi_t
\end{align*}
$$

$\ \quad\ \quad\ \quad$ 对于时间步 $t=T$, 有 $\Phi_T = -U_T$ 且 $\Psi_T = 0$.

$\ \quad\ \quad$ <u>事实 2:</u> 可以证明，最优策略是状态的线性函数。

$\ \quad\ \quad\ \quad$ 已知 $V_{t+1}^*$ 等价于已知 $\Phi_{t+1}$ 和 $\Psi_{t+1}$.

$\ \quad\ \quad\ \quad$ 因此只需要解释如何从 $\Phi_{t+1}$ 和 $\Psi_{t+1}$ 以及问题的其他参数计算 $\Phi_t$ 和 $\Psi_t$.

$$
\begin{align*}
    V_t^*(s_t) 
        &= s_t^\top \Phi_t s_t + \Psi_t \\
        &= \max_{a_t} \left[ R^{(t)}(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim P_{s_t, a_t}^{(t)}}[V_{t+1}^*(s_{t+1})] \right] \\
        &= \max_{a_t} \left[ -s_t^\top U_t s_t - a_t^\top V_t a_t + \mathbb{E}_{s_{t+1} \sim \mathcal{N}(A_t s_t + B_t a_t, \Sigma_t)}[s_{t+1}^\top \Phi_{t+1} s_{t+1} + \Psi_{t+1}] \right]
\end{align*}
$$

$\ \quad\ \quad\ \quad$ 其中第二行是最优值函数的定义。

$\ \quad\ \quad\ \quad$ 第三行是通过将模型动态过程以及二次假设代入得到的。

$\ \quad\ \quad\ \quad$ 注意到最后一个表达式是 $a_t$ 的二次函数，因此可以 (容易地) 优化。[^1] 

$\ \quad\ \quad\ \quad$ 得到最优动作 $a_t^*$

$$
\begin{align*}
    a_t^* 
      &= \left[ (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} A_t \right] \cdot s_t \\
      &= L_t \cdot s_t
\end{align*}
$$
$\ \quad\ \quad\ \quad$ 其中 $L_t := [(B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} A_t]$

$\ \quad\ \quad\ \quad$ 这是一个令人印象深刻的结果：最优策略对 $s_t$ 是**线性**的。

$\ \quad\ \quad\ \quad$ 给定 $a_t^*$, 可以求解 $\Phi_t$ 和 $\Psi_{t}$ . 最终得到 **离散里卡蒂方程 (Discrete Ricatti equations)**

$$
\begin{align*}
    &\Phi_t = A_t^\top \left( \Phi_{t+1} - \Phi_{t+1} B_t (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} \right) A_t - U_t \\
    &\Psi_t = -\text{tr}(\Sigma_t \Phi_{t+1}) + \Psi_{t+1}
\end{align*}
$$

$\ \quad\ \quad$ <u>事实 3:</u> 注意到 $\Phi_{t}$ 既不依赖于 $\Psi$ 也不依赖于噪声 $\Sigma_t$! ^fact3

$\ \quad\ \quad\ \quad$ 由于 $L_t$ 是 $A_t, B_t$ 和 $\Phi_{t+1}$ 的函数，这意味着最优策略也**不依赖于噪声**！

$\ \quad\ \quad\ \quad$ (但是 $\Psi_{t}$ 确实依赖于 $\Sigma_t$, 这意味着 $V_t^*$ 依赖于 $\Sigma_t$.)

总结一下，LQR 算法的工作方式如下：

1. (如果需要) 估计参数 $A_t, B_t, \Sigma_t$.
2. 初始化 $\Phi_T := -U_T$ 和 $\Psi_T := 0$.
3. 迭代更新 $t = T-1 \dots 0$ 时的 $\Phi_t$ 和 $\Psi_t$, 使用离散里卡蒂方程。如果存在使状态趋于零的策略，则收敛性得到保证！

利用[[chapter16_LQR_DDP_and_LQG#^fact3|事实 3]]，可以更巧妙地使算法运行得 (稍微) 快一些！由于最优策略不依赖于 $\Psi_t$, 并且 $\Phi_t$ 的更新仅依赖于 $\Phi_t$, 因此仅更新 $\Phi_t$ 就足够了！

| [[chapter15_reinforcement_learning\|上一章]] | [[CS229_CN/index#目录\|目录]] | [[chapter17_policy_gradient_REINFORCE\|下一章]] |
| :---------------------------------------: | :-----------------------: | :------------------------------------------: |

[^1]: 对 $a_t$ 求导并令导数等于零。
