---
title: ACBatch
---
# ACBatch: Adaptive and Cooperative Batching for Edge Inference

*Ziming Yang*$^1$, Zichuan Zheng$^1$, Liyou Deng$^1$, Shan Zhang$^{1\ 2\ *}$, Zhiyuan Wang$^{1\ 2}$ and Hongbin Luo$^{1\ 2}$ 
$^1$ School of Computer Science and Engineering, Beihang University, Beijing, China

$^2$ Zhongguancun Laboratory, Beijing, China

$^*$ Corresponding author

> [!abstract] 
> *Batching is a key technique in deep learning inference that enhances computational efficiency. Although widely applied in the cloud, batching may suffer from longer batch latency at edge servers due to highly dynamic task arrivals. In this paper, we propose an Adaptive and Cooperative Batching (ACBatch) framework for edge inference, wherein temporal adaptive batching and spatial task steering are jointly devised to balance the trade-off between batch latency and computational efficiency. To this end, a batch efficiency model is built to quantify the relationship between computational efficiency and batch size based on empirical measurements across diverse computing platforms and mainstream neural networks. Then, an optimization problem is formulated to minimize the completion time of a task sequence under ACBatch. For the simplified single-server case, the problem exhibits an optimal substructure and is solved by our proposed Dynamic Programming-based Adaptive Batching algorithm. For the general multi-server case, the optimization of ACBatch is proved NP-hard, and we propose the Multi-Server Cooperative Batching algorithm by iteratively optimizing batching and steering. Real-trace experiments show that ACBatch achieves an average improvement of 89.17\% in completion time and 76.52\% in latency compared to state-of-the-art methods.*

*Index Terms*—cooperative edge computing, edge inference, batching, traffic steering

## I. Introduction

Driven by the escalating demand for artificial intelligence (AI), inference has become one of the dominant tasks of edge computing \[[[ACBatch/index#^li2019edge|1]]]. Specifically, inference involves applying a trained neural network model to new data to generate predictions or insights and is expected in a wide range of practical applications. For example, in the Barcelona smart city project, over 3000 edge devices cooperatively process traffic data from about 100000 cameras to manage traffic flows, reducing congestion and enhancing safety \[[[ACBatch/index#^khan2020edge|2]]]. Similarly, in smart manufacturing, real-time edge processing supports automated quality inspections and production line adjustments, thereby improving efficiency and reducing waste \[[[ACBatch/index#^nain2022towards|3]]]. Emerging standards such as IEEE P1934 for OpenFog Reference Architecture and ETSI Multi-access Edge Computing (MEC) specifications guide the development and deployment of edge technologies \[[[ACBatch/index#^antonini2019fog|4]], [[ACBatch/index#^giust2017multi|5]]].

Batching is a key technological approach to enhance inference computation efficiency, widely applied in cloud-based AI applications \[[[ACBatch/index#^olston2017tensorflow|6]], [[ACBatch/index#^nvidia2022triton|7]], [[ACBatch/index#^moritz2018ray|8]]]. Through combining a batch of input tensors of the same size into a higher-dimensional tensor, inference with batching can fully exploit the parallel tensor operation capabilities of accelerators (such as GPUs and TPUs) \[[[ACBatch/index#^nvidia2018ai|9]], [[ACBatch/index#^jouppi2023tpu|10]], [[ACBatch/index#^nvidia2020jetson|11]]]. Batching offers higher computational efficiency than sequentially processing the input data, thereby reducing overall latency and enhancing throughput \[[[ACBatch/index#^olston2017tensorflow|6]], [[ACBatch/index#^nvidia2022triton|7]]]. For instance, batching can enhance server computational efficiency by over 100\% compared to those without batching \[[[ACBatch/index#^inoue2021queuing|12]]]. Extensive work has been conducted on batching within cloud computing environments, wherein schemes of presetting static batch sizes are designed \[[[ACBatch/index#^olston2017tensorflow|6]], [[ACBatch/index#^nvidia2022triton|7]], [[ACBatch/index#^moritz2018ray|8]]].

As edge servers are rather resource-constrained, computation-efficient batching becomes even more critical when implementing inference applications. However, existing studies on cloud batching often fall short when applied to edge computing, primarily due to the *dynamics* of task arrivals at edge nodes \[[[ACBatch/index#^abbas2017mobile|13]]]. Unlike cloud servers, which experience relatively stable and high-density task arrivals, edge nodes face significantly lower and more sporadic task arrivals due to the spatially constrained coverage. Accordingly, edge servers usually wait for a longer time to form computation-efficient batches, compromising the timeliness of edge computing.

![[scenario.svg|500]] ^fig1
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 1: Illustration of an intelligent surveillance system with ACBatch in a smart city.</p>

In this paper, we present an Adaptive and Cooperative Batching (ACBatch) framework to achieve low-latency edge inference. Specifically, we answer the two following key questions:

> [!question]- **Question 1.** *How to balance computational efficiency and batch waiting time over dynamic task arrivals?*
> An effective strategy involves optimizing the completion time of dynamic task arrival sequences through the adaptive adjustment of batch sizes. Therefore, it is essential to quantify the relationship among batch size, computational efficiency, and batch waiting time. To this end, we measure batch efficiency across diverse computing platforms and popular edge inference neural networks. The results show that batch efficiency increases sub-linearly with batch size. We introduce a general model to describe this relationship quantitatively. Furthermore, we observe a sequential correlation between tasks and batches on batch size optimization and thus propose the Dynamic Programming-based Adaptive Batching (DPAB) algorithm to minimize the completion time of a task sequence with low complexity. DPAB mainly provides a solution to fit the traffic dynamics in the temporal domain. Additionally, the multi-access of edge servers motivates us to rethink the problem in the spatial domain, which leads to the second question.

> [!question]- **Question 2.** *How to mitigate the influence of traffic dynamics and enhance the overall performance from the network aspect?*
> By leveraging the multi-access technologies, a mobile user can be flexibly steered to other surrounding access points and edge servers for service. Thus, we can steer and concentrate the closely arrived tasks at specific edge servers, whereby the waiting time for batching at each edge server can be reduced without compromising computation efficiency. Following this idea, we jointly optimize adaptive batching and cooperative traffic steering to enhance the performance of ACBatch. The formulated problem is proved to be NP-hard, and we propose a heuristic algorithm named Multi-Server Cooperative Batching (MSCB) to solve the problem.

Our main results and key contributions are summarized as follows:

* *Novel Problem Formulation:* We introduce a batch efficiency model to quantify the relationship between computing efficiency and batch size based on the measurement results of diverse computing platforms and neural networks. Then, the ACBatch framework is proposed and optimized to minimize the completion time of a task sequence. To our knowledge, this is the first study on the joint design of adaptive batching and cooperative traffic steering for edge inference.
* *Efficient Batching Scheme Design:* We start from the simplified single-server case and propose the DPAB algorithm, which adaptively optimizes the batch size to fit the dynamic inference task arrivals. Extending DPAB to multiple servers, we incorporate spatial task steering and propose the heuristic iterative algorithm MSCB. MSCB optimizes batching and steering jointly for the general multi-server environment, achieving low complexity and performance guarantee.
* *Performance Verification:* Real-trace experimental results demonstrate an average performance improvement of 89.17\% in completion time and 76.52\% in latency compared to state-of-the-art baselines. Empirical results also show that ACBatch effectively handles edge computing scenarios with extensive workloads, significant burstiness, and high spatial aggregation. In addition, ablation studies reveal that our joint optimization of batching and steering enhances the computational efficiency to 1.24x, compared to an ACBatch variation without steering, while adding the waiting time by 0.11x.

The rest of this paper is organized as follows. Section [[ACBatch/index#II. RELATED WORK|II]] introduces related studies. Section [[ACBatch/index#III. SYSTEM MODEL|III]] describes the system model, and Section [[ACBatch/index#III. SYSTEM MODEL|IV]] formulates the optimization problem. In Section [[ACBatch/index#V. ACBATCH OPTIMIZATION|V]], we present the Dynamic Programming-based Adaptive Batching algorithm for the single-server case. Next, we extend it to the Multi-Server Cooperative Batching algorithm to solve the general multi-server case. Experimental results are shown in Section [[ACBatch/index#VI. PERFORMANCE EVALUATIONS|VI]]. We conclude this paper in Section [[ACBatch/index#VII. CONCLUSION|VII]].

## II. Related Work

### A. Static Batching

Some studies (e.g., \[[[ACBatch/index#^nvidia2018ai|9]]]) focus on static batching for cloud environments under stable workloads, where a single batch size is predetermined for all tasks. NVIDIA \[[[ACBatch/index#^nvidia2018ai|9]]] employs batch sizes of 16 or 32 under long-term stable workloads to enhance throughput while incurring minimal latency additions. Nevertheless, static batching falls short when facing dynamic workloads.

### B. Dynamic Batching

Some studies \[[[ACBatch/index#^zhang2022batch|14]], [[ACBatch/index#^ali2020batch|15]], [[ACBatch/index#^lu2018crowdvision|16]]] investigate dynamically adjusting batch size under dynamic workloads based on different levels of task arrival information. In the absence of any prior information about task arrivals, Zhang et al. in \[[[ACBatch/index#^zhang2022batch|14]]] utilize a reinforcement learning approach to predict future task arrival patterns and adjust batch sizes accordingly. When task arrival patterns can be determined, such as Poisson processes and Markov processes, Ali et al. in \[[[ACBatch/index#^ali2020batch|15]]\] model the probability distribution of latency concerning batch size and search for optimal batch sizes to minimize latency or cost. When accurate arrival times are known, Lu et al. in \[[[ACBatch/index#^lu2018crowdvision|16]]\] develop a SplitShift method for batch allocation. However, they greedily split batches to minimize waiting time, which achieves a sub-optimal trade-off between computational efficiency and waiting time and damages the overall performance.

### C. Batching with Task Steering

Some studies \[[[ACBatch/index#^cui2020e2bird|17]], [[ACBatch/index#^zhang2023bcedge|18]]] focus on integrating task steering with static batching. They treat batching as a limiting condition within their problem formulations. For example, Cui et al. in \[[[ACBatch/index#^cui2020e2bird|17]]] propose to steer tasks to different devices with preset fixed batch sizes to balance the remaining memory size. Similarly, Zhang et al. in \[[[ACBatch/index#^zhang2023bcedge|18]]] employ reinforcement learning to decide on task steering to instances with static batch configurations to maximize throughput.

Recent studies (e.g., \[[[ACBatch/index#^zhang2019edgebatch|19]]]) have explored the coordination of dynamic batching and task steering across multiple edge servers. Zhang et al. in \[[[ACBatch/index#^zhang2019edgebatch|19]]] employ a stochastic approach for optimal task batching based on a probability model, using an online regret minimization method, and address task steering through a warehouse model. However, they do not jointly optimize batching and steering. Instead, they define a utilization threshold for each server based on static parameters that estimate batch efficiency, directing tasks to servers only when this threshold is not exceeded. This approach limits the adaptability and effectiveness of task distribution in dynamic edge environments.

Our proposed ACBatch framework for edge inference also coordinates dynamic batching and task steering and **differs from the above studies in two aspects**. First, we introduce a general batch efficiency model that quantifies the relationship between efficiency and batch size, enabling precise computation of batching costs and determination of optimal batch settings. Second, we optimize adaptive batching and steering jointly. This approach allows batching to dynamically adjust batch sizes based on the distribution of steered tasks, while steering strategies are refined in response to temporal changes post-batching.

## III. System Model

<p style="text-align: center; margin-bottom: .35em; font-size: 0.9em; opacity: 0.8;">TABLE I: Key notations.</p>

|            Symbol            | Description                                               |
| :--------------------------: | --------------------------------------------------------- |
|       $\mathcal{M}, M$       | Set and number of edge servers                            |
|         $\eta_{m,b}$         | Batching efficiency of server $m$ with batch size $b$     |
|        $\tau_{m1,m2}$        | Transmission latency from server $m1$ to server $m2$      |
|       $\mathcal{N}, N$       | Set and number of inference tasks                         |
|        $s_{n}, t_{n}$        | Host server and arrival time of task $n$                  |
|       $\mathcal{K}, K$       | Set and number of batch indices                           |
|    $\mathcal{I}, I_{n,k}$    | Indicator matrix and indicator of task $n$ in batch $k$   |
|    $\mathcal{J}, J_{k,m}$    | Indicator matrix and indicator of batch $k$ on server $m$ |
|           $b_{k}$            | Size of batch $k$                                         |
|           $m_{k}$            | Processing server of batch $k$                            |
| $q_{k}, r_{k}, p_{k}, e_{k}$ | Start, ready, processing and end time of batch $k$        |
|             $k'$             | Previous batch of batch $k$ on server $m_k$               |


In this section, we introduce our measurement-based batch efficiency model. With a detailed task model of an edge inference application, we propose our ACBatch framework and provide an overview.

### A. Measurement-based Batch Efficiency Model

The time required to process a batch task with batch size $b$ on a specific server $m$ is represented by $c_{m,b}$. The batching efficiency can thus be given by $\eta_{m,b} = { b c_{m,1}}/{c_{m,b}}$, which is platform-dependent and varies with different batch sizes \[[[ACBatch/index#^nvidia2018ai|9]]].

We illustrate the measurement of batching efficiency across different hardware platforms \[[[ACBatch/index#^nvidia2018ai|9]], [[ACBatch/index#^nvidia2020jetson|11]]] and popular neural networks \[[[ACBatch/index#^he2016deep|20]], [[ACBatch/index#^simonyan2014very|21]], [[ACBatch/index#^szegedy2015going|22]]] in [[ACBatch/index#^fig2|Fig. 2]]. Note that the maximum permissible batch sizes on different platforms are constrained by memory limitations and the neural networks adopted. For example, in our measurement, the maximum batch size on the NVIDIA Tesla V100 is about 128, whereas, on the NVIDIA Jetson AGX Xavier, it is limited to 32 or 16.  

![[batch_eff.png|500]] 
^fig2
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 2: Batch efficiency measurements on different hardware and neural networks.</p>

To quantify the computational efficiency of batching, we model the relationship between batching efficiency and batch sizes by fitting the measurement results. We find that batching efficiency $\eta_{m,b}$ sub-linearly increases with batch size $b$ and can be well represented by the following general form:

^eq1
$$
\eta_{m,b} = f_m(b)=\alpha_m\log{(b)} + \theta_m,\tag{1}
$$

where $\alpha_m$ and $\theta_m$ are platform-specific parameters influenced by the utilized hardware and software. The parameter $\theta_m$ is generally associated with the hardware capability to handle small-scale matrix operations \[[[ACBatch/index#^nvidia2018ai|9]]]. Conversely, $\alpha_m$ is often related to the general efficiency of software execution \[[[ACBatch/index#^nvidia2022tensorrt|23]]]. As shown in [[ACBatch/index#^tab2|Table II]], our batch efficiency model achieves at least $0.983$ of goodness of fit, indicating the effectiveness of our model.

<p style="text-align: center; margin-bottom: .35em; font-size: 0.9em; opacity: 0.8;">TABLE II: Parameter values and the corresponding goodness of fit R<sup>2</sup> for the batch efficiency model.</p>

|     Hardware      | Neural Network | $\alpha$ | $\theta$ | $R^2$ |
| :---------------: | :------------: | :------: | :------: | :---: |
|                   |    ResNet50    |  2.117   |  -0.601  | 0.997 |
|    Tesla V100     |     VGG19      |  0.518   |  1.243   | 0.983 |
|                   |   GoogLeNet    |  1.743   |  -0.713  | 0.993 |
|                   |    ResNet50    |  0.203   |  1.308   | 0.992 |
| Jetson AGX Xavier |     VGG19      |  0.434   |  1.128   | 0.994 |
|                   |   GoogLeNet    |  0.179   |  1.266   | 0.983 |

^tab2

### B. Task Model

The application of edge inference for traffic image classification in a smart city \[[[ACBatch/index#^zhao2017trafficnet|24]], [[ACBatch/index#^wen2020ua|25]]] is depicted in [[ACBatch/index#^fig1|Fig. 1]]. In this scenario, cameras are strategically positioned on streets and in neighborhoods to monitor traffic flow. These cameras continuously capture traffic images and selectively transmit the most relevant ones to their host edge servers for classification, providing detailed analysis. A network of edge servers, denoted as $\mathcal{M}=\{1, ..., m, ..., M\}$, cooperatively process inference tasks, represented as $\mathcal{N} = \{1, ..., n, ..., N\}$. Task $n$ is received by a host server $s_n \in \mathcal{M}$ at time $t_n$, and can be steered from server $m_1$ to server $m_2$ with additional transmission latency $\tau_{m1,m2}$.

### C. ACBatch Framework

![[framework.svg]] ^fig3
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 3: Overview of the ACBatch Framework.</p>

We illustrate the ACBatch framework in [[ACBatch/index#^fig3|Fig. 3]], detailing its three integral components: (1) Task Pooling: Tasks of all edge servers are reorganized into a sequence based on their arrival times, providing a comprehensive timeline view; (2) Adaptive Batching: The task sequence is adaptively grouped into $K$ batches based on their arrival times and associated server nodes; (3) Cooperative Steering: Batched are allocated to servers considering each server’s service capabilities, whereby the task distribution are spatially reshaped to form more efficient batches with larger size or lower waiting time.

Let $\mathcal{K} = \{1, \dots, k, \dots, K\}$ denote the set of batch indices. $\mathcal{I}=[I_{n,k}]_{N\times K}$ is a 0-1 indicator matrix, where $I_{n,k}$ represents whether task $n$ belongs to batch $k$. The size of batch $k$ is then given by $b_k=\sum_{n\in\mathcal{N}}I_{n,k}$. $\mathcal{J}=[J_{k,m}]_{K\times M}$ represents the allocation of batches to servers, where $J_{k,m}$ is also a 0-1 indicator showing whether batch $k$ is processed on server $m$. The processing server for batch $k$ is denoted by $m_k = \arg\max_m J_{k,m}$.

The completion time $e_k$ of batch $k$ includes its processing time and start time, given by:

$$
e_k=p_k+q_k, \tag{2}
$$ 
 where $p_k$ is the processing time and $q_k$ is the start time of batch $k$. Note that:

$$
p_k = \sum_{m\in\mathcal{M}} J_{k,m}c_{m,b_k}. \tag{3}
$$

where $b_k = \sum_{n\in\mathcal{N}}I_{n,k}$ is the size of batch $k$.

The start time, $q_k$, is subject to two constraints to ensure causality. First, $q_k$ cannot precede the ready time of batch $k$, which is the moment when all tasks within batch $k$ have been transmitted to the processing server. Second, $q_k$ must be delayed until the completion of the previous batch $k'$ on the processing server. Therefore, $q_k$ is given by:

$$
q_k=\max(r_k,e_{k'}),\tag{4}
$$

where $r_k$ is the ready time of batch $k$:

$$
r_k = \max_{n\in\mathcal{N}} {I_{n,k}(t_n+ \tau_{s_n,m_k})}. \tag{5}
$$

and $m_k=\arg\max_{m} J_{k,m}$ represents the processing server of batch $k$.

The previous batch $k'$ corresponds to the batch with the largest ready time that is less than $r_k$ on server $m_k$. To identify $k'$, we define an auxiliary function to clip ready times no less than $r_k$ to zero:

$$
\text{clip}(k_1, k) =
  \begin{cases}
    J_{k_1,m_k} r_{k_1}, & \text{if } J_{k_1,m_k} r_{k_1} < r_k, \\
    0, & \text{otherwise.}
  \end{cases} \tag{6}
$$

For batch $k_1$, if it is processed on server $m_k$ and its ready time is less than $r_k$, there is $\text{clip}(k_1, k) > 0$. Therefore, $k'$ is the batch with the largest $\text{clip}(k_1, k)$, given by:

$$
k' = \arg\max_{k_1\in\mathcal{K}} \text{clip}(k_1, k). \tag{7}
$$

## IV. Problem Formulation and Analysis

In this section, we formulate an optimization problem for adaptive and cooperative batching and analyze its hardness.

### A. Problem Formulation

The optimization problem can be formulated as [[ACBatch/index#^p1|(P1)]]:

^p1
$$
\begin{align}
  \textbf{(P1) } &\ \min_{\mathcal{K},\mathcal{I},\mathcal{J}} \max_{k\in\mathcal{K}} e_k \tag{8a}\\
  \text{s.t. } &\sum_{k\in\mathcal{K}}I_{n,k}=1, \forall n\in\mathcal{N} \tag{8b} \\
  &\sum_{m\in\mathcal{M}}J_{k,m}=1, \forall k\in\mathcal{K} \tag{8c} \\
  &K\in [\![1,N]\!] \tag{8d} \\
  &I_{n,k}\in\{0,1\},\forall n\in\mathcal{N}, k\in\mathcal{K} \tag{8e} \\
  &J_{k,m}\in\{0,1\}, \forall k\in\mathcal{K}, m\in\mathcal{M} \tag{8f} \\
  &1\le \sum_{n\in\mathcal{N}}{I_{n,k}} \le \sum_{m\in\mathcal{M}}J_{k,m}B_m, \forall k\in\mathcal{K} \tag{8g}
\end{align}
$$

The object $\mathbf{(P1)}$ is to minimize the completion time of all tasks $N$, i.e., the completion time of all batches $K$. Constraint $\text{(8b)}$ makes sure a task is assigned one and only one batch. Constraint $\text{(8c)}$ means a batch is processed on one and only one server. Constraint $\text{(8d)}$ regulates that the number of batches is less than or equal to the task number, and there is at least one batch. Constraints $\text{(8e)}$ and $\text{(8f)}$ regulate $I_{n,k}$ and $J_{k,m}$ as binary indicators. Constraint $\text{(8g)}$ represents that the batch size should not exceed the maximum batch size $B_m$ of server $m$.

### B. Hardness of the Problem

We establish the NP-hardness of the problem by reducing it to the well-known parallel Batch Processing Machine (BPM) scheduling problem, which has already been proven to be NP-hard \[[[ACBatch/index#^lageweg1982computer|26]]]. Consider a simplified case when the edge servers are homogeneous (i.e., $B_m = B, \eta_{m,b} = f(b), \forall m \in \mathcal{M}$) and the traffic steering cost can be ignored (i.e., $\tau_{m_1,m_2} = 0, \forall m_1, m_2 \in \mathcal{M}$). We reduce [[ACBatch/index#^p1|(P1)]] to a parallel BPM scheduling scenario. In this scenario, there are $M$ parallel BPMs, each capable of processing up to $B$ tasks simultaneously. The batch efficiency increases monotonically with the batch size, as expressed in [[ACBatch/index#^eq1|(1)]]. Here, $N$ tasks are awaiting processing, with the objective of minimizing the make-span, which is defined as the total completion time of all tasks. This reduction proves the NP-hardness of our problem.

In problem [[ACBatch/index#^p1|(P1)]], the completion time crucially depends on the number of batches, their sizes, and the task steering across multiple edge servers. Given the number of batches $K$, there are $NK + KM$ decision variables. This results in a total of $\sum_{1 \leq K \leq N} 2^{K(N+M)}$ potential variable combinations. The computational complexity for an exhaustive search thus reaches $O(2^{N^2 + NM})$, illustrating the significant computational complexity in solving this problem.

## V. ACBatch Optimization

We start with the simple single-server case and focus on the optimization of adaptive batching. Then, we extend to the general multi-server case and jointly optimize batching and traffic steering.

### A. Dynamic Programming-based Adaptive Batching

For the single-server case, $\mathcal{M}=\{1\}$, and thus $J_{k,1}=1,\forall k\in\mathcal{K}$. We remove the subscript $m$ for simplification, and the adaptive batching problem can be formulated as [[ACBatch/index#^p2|(P2)]]:

^p2
$$
\begin{align}
  \textbf{(P2) } &\ \min_{\mathcal{K},\mathcal{I}} \max_{k\in\mathcal{K}} e_k \tag{9a}\\
  \text{s.t. } &\sum_{k\in\mathcal{K}}I_{n,k}=1, \forall n\in\mathcal{N} \tag{9b} \\
  &K\in [\![1,N]\!] \tag{9c} \\
  &I_{n,k}\in\{0,1\},\forall n\in\mathcal{N}, k\in\mathcal{K} \tag{9d} \\
  &1\le \sum_{n\in\mathcal{N}}{I_{n,k}} \le B, \forall k\in\mathcal{K} \tag{9e}
\end{align}
$$

This problem still presents an exponential solution space of $\sum_{1\le K\le N}{2^{KN}}$. To efficiently tackle this, (1) we leverage insights into batch sequentiality to significantly reduce the solution space given by [[ACBatch/index#^prop1|Proposition 1]]; and (2) we recognize the problem’s optimal substructure, essential for applying dynamic programming. Based on these insights, we develop a dynamic programming-based algorithm that computes the optimal batching scheme for given task arrivals from the solution of sub-problems, ensuring computational efficiency and optimality within the constrained solution space.

**Proposition 1.** *For problem [[ACBatch/index#^p2|(P2)]], there exists an optimal scheme $(\mathcal{K}^*, \mathcal{I}^*)$ which satisfies that $\{ (n_1, k_1, n_2, k_2) \in \mathcal{N} \times \mathcal{K}^* \times \mathcal{N} \times \mathcal{K}^* : I^*_{n_1,k_1} = I^*_{n_2,k_2} = 1, n_1 < n_2, k_1 > k_2 \} = \emptyset$.* ^prop1

> [!Proof]- 
> There is a sufficient condition to establish the proposition: if there exists an optimal scheme $(\mathcal{K}, \mathcal{I})$ which does not satisfy the condition in [[ACBatch/index#^prop1|Proposition 1]], then there also exists an optimal scheme $(\mathcal{K}^*, \mathcal{I}^*)$ satisfying the condition. To construct such an optimal scheme $(\mathcal{K}^*, \mathcal{I}^*)$, we propose to exchange the allocations of $n_1$ and $n_2$ in $(\mathcal{K}, \mathcal{I})$: assign task $n_1$ to batch $k_2$ and task $n_2$ to batch $k_1$, denoted by the superscript $*$ for the related variables post-exchanging. This exchange does not alter the sizes of batches $k_1$ and $k_2$, hence $p_{k_1}^* = p_{k_1}$ and $p_{k_2}^* = p_{k_2}$.
> 
> For batch $k_1$, if task $n_1$ is not the last in the batch, the ready time $r_{k_1}$ remains unchanged, and hence $e_{k_1}^* \le e_{k_1}$. If task $n_1$ is the last, then after the swap, task $n_2$ dictates the batch’s end time, which cannot exceed its previous end time because $e_{k_2} \ge t_{n_2}$, hence $e_{k_1}^* \le e_{k_1}$. And for batch $k_2$, because $t_{n_1} < t_{n_2}$, the earliest start time $r_{k_2}^*$ remains unchanged or is earlier, thus $q_{k_2}^* \le q_{k_2}$ and $e_{k_2}^* \le e_{k_2}$. 
> 
> In both scenarios, the adjusted scheme $(\mathcal{K}^*, \mathcal{I}^*)$ does not increase the processing time of any batch, thereby proving the proposition. <span style="float: right;">▯</span>

[[ACBatch/index#^prop1|Proposition 1]] indicates the existence of an optimal solution, where each batch consists of adjacent tasks. Accordingly, the solution space can be reduced from exponential ($\sum_{1\le K\le N}{2^{KN}}$) to combinatorial ($\sum_{1\le K\le N}{\binom{N-1}{K-1}}$) level.

Furthermore, the sequential order provides a crucial insight: the optimal solution to the original problem can be derived from the optimal solutions of its sub-problems, demonstrating the optimal sub-structure of the problem. Define $C_{min}[n]$ as the minimum cost for the subset of tasks $\mathcal{N}_n = \{1, ..., n\}$. Denote by $C[n, b]$ the minimum cost solution when the last batch of the subset of tasks $\mathcal{N}_n$ consists of $b$ tasks. Consequently, $C_{min}[n] = \min_b C[n, b]$, where $C[n, b]$ can be derived from $C_{min}[n-b]$.

![[algo1.svg|500]] ^fig4
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 4: Example of the DPAB algorithm when N=3. To get the minimal cost of the whole task set Cₘᵢₙ[3], we can compare all C[3,b] for b ranging from 1 to 3. Meanwhile, C[3,b] can be deduced from Cₘᵢₙ[3-b], and the corresponding optimal solution is red-boxed in the figure.</p>

This relationship indicates that the optimal solution for $\mathcal{N}$ can be deduced from the optimal solutions of its sub-problems for $\mathcal{N}_n, n\in \{1, ..., N-1\}$. This ensures the feasibility of using dynamic programming to determine the minimum cost for the entire task set. Thus, we propose the Dynamic Programming-based Adaptive Batching (DPAB) algorithm to address [[ACBatch/index#^p2|(P2)]], detailed in [[ACBatch/index#^algo1|Algorithm 1]] and visually depicted in [[ACBatch/index#^fig4|Fig. 4]]. The time complexity of [[ACBatch/index#^algo1|Algorithm 1]] is $O(N^2)$, consisting of $O(N)$ for initialization, $O(N^2)$ for dynamic programming, and $O(N)$ for tracing back, where $N$ represents the number of tasks.

^algo1
<div style="border-top: 2px solid; border-bottom: 1px solid;"> <b>Algorithm 1</b> Dynamic Programming-based Adaptive Batching</div>

**Input:** Arrival times $t[1...N]$ of the tasks, maximum batch size $B$, parallel efficiency function $f$ and computing time per task $c$ of the server;  
**Output:** Batch size list $Bs$;

1: $\space$ **Initialize** $C_{\text{min}}[n]\leftarrow +\infty, \forall n\in[1,N]$, $C_{\text{min}}[0]\leftarrow 0$, and $\text{Path}[0]\leftarrow 0$  
2: $\space$ **for** $n\in [1,N]$ **do**  
3: $\space\quad$ **for** $b\in [1,\min(B, n)]$ **do**  
4: $\space\quad\quad$ $C[n,b]\leftarrow\max(t[n], C_{\text{min}}[n-b])+b*c/f(b)$  
5: $\space\quad\quad$ **if** $C[n,b]<C_{\text{min}}[n]$ **then**  
6: $\space\quad\quad\quad$ $C_{min}[n]\leftarrow C[n,b]$  
7: $\space\quad\quad\quad$ $\text{Path}[n]\leftarrow n-b$  
8: $\space$ $n \leftarrow N$  
9: $\space$ **Declare** $Bs$ an empty list  
10:  **while** $n \ne 0$ **do**  
11: $\quad$ $Bs.\texttt{append}(n-\mathrm{Path}[n])$  
12: $\quad$ $n \leftarrow \mathrm{Path}[n]$  
13:  **return** $Bs$

<hr style="
    border: 0;
    border-top: 1px solid;
">

### B. Multi-Server Cooperative Batching

In this section, we introduce our Multi-Server Cooperative Batching (MSCB) algorithm to solve the problem [[ACBatch/index#^p1|(P1)]] and analyze its computational complexity and performance guarantee.

We extend our single-server adaptive batching algorithm to a Multi-Server Cooperative Batching algorithm by utilizing an iterative heuristic approach. In particular, MSCB consists of two processes: (1) adaptive batching to optimize the batching strategies of each server and (2) task steering to adjust the task distribution for more efficient batch processing. Firstly, DPAB is conducted to fit the dynamic task arrivals for each server individually. Then, task steering is executed to reshape the traffic distribution in the spatial domain. We iterate between batching and steering until the task assignments $\mathcal{I}$ remain unchanged post-batching.

The key design of task steering encompasses two main aspects. (1) **Task candidate set pruning**: Initially, the task candidate set includes all tasks. We begin by reducing this set to tasks on the server with the largest completion time, denoted as $m^\text{src}$. Steering these tasks can immediately decrease the completion time. Subsequently, we further refine the candidate set to include only the last task of each batch on $m^\text{src}$, following the guidelines outlined in [[ACBatch/index#^prop2|Proposition 2]]. (2) **Greedy task selection**: Tasks within the candidate set are greedily selected based on their reduction in completion time.

**Proposition 2.** For all task $n$ in the set $\mathcal{N}^{(k)}$, it holds that: ^prop2

$$
C^\text{src}_{n^\text{k\_last}} \leq C^\text{src}_n,
$$

where $\mathcal{N}^{(k)}$ is the set of all tasks $n$ satisfying $I_{n,k}=1$ and $n^\text{k\_last} = \arg\max_{n\in\mathcal{N}^{(k)}} (t_n+\tau_{s_n,m^\text{src}})$ denotes the last task of $\mathcal{N}^{(k)}$. Denote by $C^\text{src}_n$ the completion time of the source server after steering task $n$ to the destination.

> [!Proof]-
> 
> For a batch $k$ including tasks $\{i-b+1,...,i\}$ with batch size of $b$, we denote steering $i$ and $j$ ($i-b<j<i$) with superscripts $^{(i)}$ and $^{(j)}$. We observe that $r_k^{(i)}=t_{i-1}<r_k^{(j)}=t_{i}$ and $p_k^{(i)}=\eta_{b-1}(b-1)c=p_k^{(j)}$. Therefore, $q_k^{(i)}\le q_k^{(j)}$ and consequently, $e_k^{(i)}\le e_k^{(j)}$. Since subsequent batches remain unchanged, the completion time after steering task $i$ is not longer than for $j$, i.e., $C^\text{src}_{i} \le C^\text{src}_j$, thereby proving the proposition. <span style="float: right;">▯</span>

This approach ensures that a maximum of $K$ tasks are considered for steering, where $K$ represents the total number of batches.

The proposed algorithm is summarized in [[ACBatch/index#^algo2|Algorithm 2]], and its time complexity is $O(TN^2)$ across $T$ iterations, composed of the following parts. Evaluating the steering candidates is $O(3KN)$, considering $3$ potential options---(1) creating a new batch; (2) joining the previous batch; (3) joining subsequent---for up to $K$ tasks, each requiring $O(N)$ time. Both the initial batching and subsequent re-batching processes involve a complexity of $O(N^2)$, with $N$ representing the total number of tasks.

^algo2
<div style="border-top: 2px solid; border-bottom: 1px solid;"> <b>Algorithm 2</b> Multi-Server Cooperative Batching</div>

**Input:** Arrival times $t[1...N]$ and arrived edge server $s[1...N]$ of the tasks, edge servers $E[1...M]$, transmission cost matrix $\tau[1...M][1...M]$;
**Output:** Batch size lists $Bs$ of each server, Server number which each task is assigned $As$;

1: $\space$ **Initialize** $Costs$ as the cost list of each server  
2: $\space$ $batch\_flag \leftarrow \textbf{true}$   
3: $\space$ **while** $batch\_flag$ **do**  
4: $\space\quad$ $min\_cost \leftarrow \min(Costs)$  
5: $\space\quad$ $src \leftarrow \arg\min(Costs)$  
6: $\space\quad$ Let $Cand$ contain the last task of each batch on $src$  
7: $\space\quad$ Iterate from $Cand$, try steering to other servers  
8: $\space\quad$ Compare the cost of $3$ opinions:  
9: $\space\quad\quad$ 1. creating a new batch  
10: $\quad\!\quad$ 2. joining the previous one  
11: $\quad\!\quad$ 3. joining the next one  
12: $\quad$ **if** minimal cost of these opinions $< min\_cost$ **then**  
13: $\quad\quad$ Accept the opinion with minimal cost  
14: $\quad\quad$ Update $Bs, As$  
15: $\quad$ Conduct $\texttt{DPAB}$ on each server  
16: $\quad$ **if** no gains from $\texttt{DPAB}$ **then**  
17: $\quad\quad$ $batch\_flag \leftarrow \textbf{false}$  
18: $\space$ **return** $Bs, As$

<hr style="
    border: 0;
    border-top: 1px solid;
">


We analyze the performance guarantee of MSCB in [[ACBatch/index#^prop3|Proposition 3]].

**Proposition 3.** MSCB achieves the following performance bound for the scenarios with identical servers and no transmission cost: ^prop3

$$
\frac{\max_k \hat{e}_k-t_n}{\max_ke_k^*-t_n}\le \epsilon\eta_{B}B,
$$

where $\max_k\hat{e}_k$ represents the results by MSCB, and $\max_ke_k^*$ the optimal. Denote by $\epsilon$ the length of the continuous batch including the last batch, and $B$ the maximum batch size.

> [!Proof.]-
> Consider the last steering step, with a potential re-batching after, thus setting the performance lower bound for MSCB. The completion time is the maximum of either the source or destination server’s completion time.
> 
> Let the identical servers share the same computing time $c$. The continuous $\epsilon$ batches create a period of at most $\epsilon\eta_{B}Bc$ before steering. Steering a task from the source leads to an upper limit of  $\hat{e}_\text{src}$, reducing size by one for a batch among $\epsilon$: $\hat{e}_\text{src}\le t_{n}+(\epsilon-1)\eta_{B}Bc+\eta_{B-1}(B-1)c$. While receiving a task at the destination leads to an upper limit of $\hat{e}_\text{dst}$, adding a new batch: $\hat{e}_\text{dst}\le t_n+\epsilon\eta_{B}Bc+c$.
> 
> Furthermore, the steering is accepted when the performance improves. Then we have, $\max_k\hat{e}_k\le\min(t_n+\epsilon\eta_{B}Bc, \max(\hat{e}_\text{src}, \hat{e}_\text{dst}))=t_n+\epsilon\eta_{B}Bc$.
> 
> For the optimal scenario, the earliest start for the last batch is at $t_n$, with a minimum processing time $c$ (size $1$ for the last batch), i.e., $\max_ke_k^*\ge t_n+c$.
> 
> Hence, the proposition is proved. <span style="float: right;">▯</span>

## VI. Performance Evaluations

In this section, we conduct a thorough performance evaluation of ACBatch. First, we compare its real-trace performance against state-of-the-art baselines. Next, ACBatch is assessed under varying arrival rates, burstiness, and spatial aggregation degrees. Furthermore, we investigate our steering method through ablation studies.

<p style="text-align: center; margin-bottom: .35em; font-size: 0.9em; opacity: 0.8;">TABLE III: Specifications of edge servers.</p>

| Server No. |       Type        | Transmission Latency<br>to Servers (ms) |
| :--------: | :---------------: | --------------------------------------: |
|     1      |    Tesla V100     |                       \[0, 10, 10, 15\] |
|     2      | Jetson AGX Xavier |                       \[10, 0, 10, 15\] |
|     3      | Jetson AGX Xavier |                       \[10, 10, 0, 15\] |
|     4      | Jetson AGX Xavier |                       \[15, 15, 15, 0\] |

^tab3

### A. Experiments Setup

Our evaluation focuses on a prevalent edge inference application: image classification. For this purpose, we employ ResNet50 \[[[ACBatch/index#^he2016deep|20]]], a widely recognized image classification model. The task requests are sourced from the ImageNet-22K dataset \[[[ACBatch/index#^deng2009imagenet|27]]] with an image resolution of 224×224×3, simulating real-world images captured by cameras.

The evaluation is set in a cooperative edge computing environment comprising four edge servers, tabulated in [[ACBatch/index#^tab3|Table 3]]. These servers are outfitted with popular edge processors, namely the NVIDIA Jetson AGX Xavier and the NVIDIA Tesla V100. We set a series of typical values reported in \[[[ACBatch/index#^cai2021latency|28]]] to simulate the network transmission latency among the edge servers.

The evaluation metric—completion time—is calculated as the difference between the end time and the arrival time of the last task within the input task sequence to avoid the impact of randomness in task arrivals. The input task sequence will be detailed in the following subsections.

The following baselines are selected from recent literature because they take both batching and traffic steering into consideration.

* **EdgeBatch**\[[[ACBatch/index#^zhang2019edgebatch|19]]]: a collaborative intelligent edge computing framework consisting of a stochastic task batching mechanism and a dynamic task offloading scheme. Consistent with their bootstrapping phase, we initiate the batching parameters with a 100-time-slot setup.
* **SplitShift**\[[[ACBatch/index#^lu2018crowdvision|16]]]: a batching and offloading scheme designed to balance local processing time and communication latency. Note that SplitShift offloads tasks to the cloud. To maintain consistency in our experiments that include SplitShift, the offloading latency to the cloud is standardized at 50 microseconds.

### B. Real-trace Performance Comparison

![[real_trace.png|500]] 
^fig5
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 5: Real-trace task arrivals of each time slot.</p>

We evaluate the real-trace performance of ACBatch using a task arrival dataset obtained from real-world traces \[[[ACBatch/index#^weng2022mlaas|29]]], as depicted in [[ACBatch/index#^fig5|Fig. 5]]. In this evaluation, each time slot is set to 4000 milliseconds, and the task arrivals within each time slot are treated as a separate input task sequence for one experiment.

In Fig. \ref{fig:rtexps}\subref{fig:rt_completion}, ACBatch demonstrates the shortest completion time and latency relative to the baseline approaches. Specifically, ACBatch exhibits a significant reduction in completion time by 89.17\% on average compared to EdgeBatch. Although ACBatch is not explicitly designed to optimize latency, it also shows a substantial improvement in latency performance, achieving an improvement of 76.52\% in latency compared to SplitShift, as shown in Fig. \ref{fig:rtexps}\subref{fig:rt_latency}. Fig. \ref{fig:rt_ovhd} shows that the runtime overhead of ACBatch is moderate, remaining below 10 milliseconds even in large-scale scenarios with over 1000 task arrivals, demonstrating its suitability for real-time inference applications. These results underscore the effectiveness of ACBatch in handling dynamic, real-world task arrivals.

### C. Impact of Task Arrival Patterns

We explore the effects of different task arrival patterns via a series of experiments. In these experiments, tasks follow Poisson arrival patterns. Unless specified otherwise, a number of 800 tasks are uniformly distributed across the four servers with an arrival rate ($\lambda$) of 100 tasks per second. We conduct the experiments 40 times and gather the results.

![[arrival_rate.png|500]] 
^fig8
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 8: Completion time versus arrival rates.</p>

[[ACBatch/index#^fig8|Fig. 8]] depicts the impact of arrival rates, which vary from 25 to 200 tasks per second. ACBatch achieves the lowest completion time at all arrival rates. At arrival rates under 100 tasks per second, ACBatch maintains a stable performance of approximately 30ms, demonstrating that ACBatch effectively utilizes batching to enhance efficiency. At arrival rates exceeding 100 tasks per second, the completion time for ACBatch increases slowly with the arrival rates.

![[burstiness.png|500]] 
^fig9
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 9: Completion time versus burstiness.</p>

[[ACBatch/index#^fig9|Fig. 9]] illustrates the impact of burstiness, which ranges from 20\% to 90\%. Burstiness is quantified as the ratio of tasks arriving in the final 20\% of the time period. The desired burstiness levels are achieved by varying the arrival rates during the initial 80\% of the time period and the last 20\%. ACBatch consistently delivers the best performance across all burstiness levels, though completion time rises as burstiness increases. Notably, ACBatch produces comparable results under conditions of 200 tasks per second (240.92ms) and 50\% burstiness (302.80ms), despite the latter scenario averaging 100 tasks per second. This indicates that burstiness has a more significant impact on ACBatch’s performance.

![[aggregation.png]] ^fig10
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 10: Completion time versus spatial aggregation degrees.</p>

[[ACBatch/index#^fig10|Fig. 10]] shows the impact of spatial aggregation, spanning from 25.0\% to 87.5\%. Spatial aggregation is defined as the proportion of tasks arriving on a specific server, with the remainder distributed uniformly across other servers. [[ACBatch/index#^fig10|Fig. 10(a)]] and [[ACBatch/index#^fig10|Fig. 10(b)]] show the performance of Server 1 (Tesla V100) and Server 2 (Jetson AGX Xavier) under varying aggregation degrees. ACBatch consistently outperforms other approaches across all aggregation degrees on both servers. On Server 1, the completion time initially decreases and then increases with higher aggregation, indicating that assigning more tasks to a more efficient server up to its capacity improves performance.

In conclusion, ACBatch excels under diverse arrival patterns, notably outperforming state-of-the-art methods in high-load, bursty, and unbalanced edge computing scenarios.

### D. Impact of Steering

Following the experimental settings outlined in Section [[ACBatch/index#C. Impact of Task Arrival Patterns|6.3]], we investigate the impact of steering on server workloads and evaluate its effectiveness in balancing waiting time with computational efficiency.

![[workload.png|500]] 
^fig11
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 11: Server workloads before and after steering.</p>

[[ACBatch/index#^fig11|Fig. 11]] contrasts the workloads of edge servers before and after steering. We split the task arrival period into 10 equal time slots and normalize the workload values by dividing the number of tasks by computational efficiency. The results visually confirm that our steering method effectively matches appropriate workloads with edge servers.

![[tradeoff.png|500]] ^fig12
<p style="text-align: center; margin-top: .35em; font-size: 0.9em; opacity: 0.8;">Fig. 12: Trade-off between waiting time and computational efficiency across different arrival rates (smaller markers indicate lower arrival rates).</p>

[[ACBatch/index#^fig12|Fig. 12]] compares ACBatch, ACBatch without steering, and the baseline EdgeBatch across varying arrival rates from 50 to 150 tasks per second. At higher arrival rates, ACBatch enhances computational efficiency to 1.24 times with an increase in waiting time by only 0.11 times, compared to the variation without steering. Notably, at lower arrival rates, ACBatch employs a significantly different strategy compared to the baseline EdgeBatch. ACBatch generally maintains lower waiting times, albeit at the cost of some efficiency. In contrast, EdgeBatch focuses on enhancing efficiency, which results in considerably longer waiting times. This disparity in strategy and the performance comparison highlights that ACBatch effectively balances waiting time with computational efficiency.

## VII. Conclusion

In this paper, we propose ACBatch, an innovative framework for cooperative edge inference. Specifically, ACBatch jointly optimizes temporal adaptive batching and spatial cooperative steering to balance the trade-off between batch latency and computational efficiency. The optimization problem exhibits an exponential solution space. We first analyze the sequentiality of single-server batching optimization and thus propose a dynamic programming-based algorithm. This algorithm is then extended to multi-server scenarios through iterative batching and steering optimization. We prove that the algorithm has polynomial time complexity and provides performance guarantees. Real-trace experiments demonstrates that ACBatch has a significant reduction in completion time and latency in comparison to state-of-the-art baselines. Further expeimental results confirm that ACBatch excels in high-load, bursty, and unbalanced edge computing scenarios.

---

## References

\[1\] E. Li, L. Zeng, Z. Zhou, and X. Chen, “Edge ai: On-demand accelerating deep neural network inference via edge computing,” IEEE Transactions on Wireless Communications, vol. 19, no. 1, pp. 447–457, 2019. ^li2019edge

\[2\] L. U. Khan, I. Yaqoob, N. H. Tran, S. A. Kazmi, T. N. Dang, and C. S. Hong, “Edge-computing-enabled smart cities: A comprehensive survey,” IEEE Internet of Things Journal, vol. 7, no. 10, pp. 10200–10232, 2020. ^khan2020edge

\[3\] G. Nain, K. Pattaanak, and G. Sharma, “Towards edge computing in intelligent manufacturing: Past, present and future,” Journal of Manufacturing Systems, vol. 62, pp. 588–611, 2022. ^nain2022towards

\[4\] M. Antonini, M. Vecchio, and F. Antonelli, “Fog computing architectures: A reference for practitioners,” IEEE Internet of Things Magazine, vol. 2, no. 3, pp. 19–25, 2019. ^antonini2019fog

\[5\] F. Giust, X. Costa-Perez, and A. Reznik, “Multi-access edge computing: An overview of etsi mec isg,” IEEE 5G Tech Focus, vol. 1, no. 4, p. 4, 2017. ^giust2017multi

\[6\] C. Olston, N. Fiedel, K. Gorovoy, J. Harmsen, L. Lao, F. Li, V. Rajashekhar, S. Ramesh, and J. Soyke, “Tensorflow-serving: Flexible, high-performance ml serving,” arXiv preprint arXiv:1712.06139, 2017. ^olston2017tensorflow

\[7\] NVIDIA, “Nvidia triton inference server - scheduling and batching,” Online, 2022, https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html. ^nvidia2022triton

\[8\] P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, E. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan et al., “Ray: A distributed framework for emerging {AI} applications,” in 13th USENIX symposium on operating systems design and implementation (OSDI 18), 2018, pp. 561–577. ^moritz2018ray

\[9\] NVIDIA, “Nvidia Ai Inference Platform Technical Overview,” Online, 2018, https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/t4-inference-print-update-inference-tech-overview-final.pdf. ^nvidia2018ai

\[10\] N. Jouppi, G. Kurian, S. Li, P. Ma, R. Nagarajan, L. Nai, N. Patil, S. Subramanian, A. Swing, B. Towles et al., “Tpu v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings,” in Proceedings of the 50th Annual International Symposium on Computer Architecture, 2023, pp. 1–14. ^jouppi2023tpu

\[11\] NVIDIA, “Jetson AGX Xavier Series,” Online, 2020, https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/. ^nvidia2020jetson

\[12\] Y. Inoue, “Queuing analysis of gpu-based inference servers with dynamic batching: A closed-form characterization,” Performance Evaluation, vol. 147, p. 102183, 2021. ^inoue2021queuing

\[13\] N. Abbas, Y. Zhang, A. Taherkordi, and T. Skeie, “Mobile edge computing: A survey,” IEEE Internet of Things Journal, vol. 5, no. 1, pp. 450–465, 2017. ^abbas2017mobile

\[14\] L. Zhang, Y. Zhang, X. Wu, F. Wang, L. Cui, Z. Wang, and J. Liu, “Batch adaptive streaming for video analytics,” in IEEE INFOCOM 2022-IEEE Conference on Computer Communications. IEEE, 2022, pp. 2158–2167. ^zhang2022batch

\[15\] A. Ali, R. Pinciorelli, F. Yan, and E. Smirni, “Batch: Machine learning inference serving on serverless platforms with adaptive batching,” in SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2020, pp. 1–15. ^ali2020batch

\[16\] Z. Lu, K. Chan, S. Pu, and T. La Porta, “Crowdvision: A computing platform for video crowdprocessing using deep learning,” IEEE Transactions on Mobile Computing, vol. 18, no. 7, pp. 1513–1526, 2018. ^lu2018crowdvision

\[17\] W. Cui, Q. Chen, H. Zhao, M. Wei, X. Tang, and M. Guo, “E2bird: Enhanced elastic batch for improving responsiveness and throughput of deep learning services,” IEEE Transactions on Parallel and Distributed Systems, vol. 32, no. 6, pp. 1307–1321, 2020. ^cui2020e2bird

\[18\] Z. Zhang, H. Li, Y. Zhao, C. Lin, and J. Liu, “Bcedge: Slow-aware dnn inference services with adaptive batching on edge platforms,” arXiv preprint arXiv:2305.01519, 2023. ^zhang2023bcedge

\[19\] D. Zhang, N. Vance, Y. Zhang, M. T. Rashid, and D. Wang, “Edgebatch: Towards ai-empowered optimal task batching in intelligent edge systems,” in 2019 IEEE Real-Time Systems Symposium (RTSS). IEEE, 2019, pp. 366–379. ^zhang2019edgebatch

\[20\] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778. ^he2016deep

\[21\] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014. ^simonyan2014very

\[22\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1–9. ^szegedy2015going

\[23\] NVIDIA, “Nvidia TensorRT Documentation,” Online, 2022, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html. ^nvidia2022tensorrt

\[24\] D. Zhao, Y. Guo, and Y. J. Jia, “Trafficnet: An open naturalistic driving scenario library,” in 2017 IEEE 20th International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2017, pp. 1–8. ^zhao2017trafficnet

\[25\] L. Wen, D. Du, Z. Cai, Z. Lei, M.-C. Chang, H. Qi, J. Lim, M.-H. Yang, and S. Lyu, “Ua-detrac: A new benchmark and protocol for multi-object detection and tracking,” Computer Vision and Image Understanding, vol. 193, p. 102907, 2020. ^wen2020ua

\[26\] B. Lageweg, J. K. Lenstra, E. Lawler, and A. R. Kan, “Computer-aided complexity classification of combinatorial problems,” Communications of the ACM, vol. 25, no. 11, pp. 817–822, 1982. ^lageweg1982computer

\[27\] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255. ^deng2009imagenet

\[28\] Y. Cai, L. Ran, J. Zhang, and H. Zhu, “Latency optimization for d2d-enabled parallel mobile edge computing in cellular networks,” EURASIP Journal on Wireless Communications and Networking, vol. 2021, no. 1, p. 133, 2021. ^cai2021latency

\[29\] Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin, and Y. Ding, “MLaaS in the wild: Workload analysis and scheduling in Large-Scale heterogeneous GPU clusters,” in 19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22), 2022, pp. 945–960. ^weng2022mlaas