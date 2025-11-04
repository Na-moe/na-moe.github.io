---
title: "ACBatch: Adaptive and Cooperative Batching for Edge Inference"
---
*Ziming Yang*$^1$, Zichuan Zheng$^1$, Liyou Deng$^1$, Shan Zhang$^{1\ 2\ *}$, Zhiyuan Wang$^{1\ 2}$ and Hongbin Luo$^{1\ 2}$ 
$^1$ School of Computer Science and Engineering, Beihang University, Beijing, China

$^2$ Zhongguancun Laboratory, Beijing, China

$^*$ Corresponding author

*Abstract: Batching is a key technique in deep learning inference that enhances computational efficiency. Although widely applied in the cloud, batching may suffer from longer batch latency at edge servers due to highly dynamic task arrivals. In this paper, we propose an Adaptive and Cooperative Batching (ACBatch) framework for edge inference, wherein temporal adaptive batching and spatial task steering are jointly devised to balance the trade-off between batch latency and computational efficiency. To this end, a batch efficiency model is built to quantify the relationship between computational efficiency and batch size based on empirical measurements across diverse computing platforms and mainstream neural networks. Then, an optimization problem is formulated to minimize the completion time of a task sequence under ACBatch. For the simplified single-server case, the problem exhibits an optimal substructure and is solved by our proposed Dynamic Programming-based Adaptive Batching algorithm. For the general multi-server case, the optimization of ACBatch is proved NP-hard, and we propose the Multi-Server Cooperative Batching algorithm by iteratively optimizing batching and steering. Real-trace experiments show that ACBatch achieves an average improvement of 89.17\% in completion time and 76.52\% in latency compared to state-of-the-art methods.*

*Index Terms*—cooperative edge computing, edge inference, batching, traffic steering

## I. INTRODUCTION

Driven by the escalating demand for artificial intelligence (AI), inference has become one of the dominant tasks of edge computing \[[[ACBatch/index#^li2019edge|1]]\]. Specifically, inference involves applying a trained neural network model to new data to generate predictions or insights and is expected in a wide range of practical applications. For example, in the Barcelona smart city project, over 3000 edge devices cooperatively process traffic data from about 100000 cameras to manage traffic flows, reducing congestion and enhancing safety \cite{khan2020edge}. Similarly, in smart manufacturing, real-time edge processing supports automated quality inspections and production line adjustments, thereby improving efficiency and reducing waste \cite{nain2022towards}. Emerging standards such as IEEE P1934 for OpenFog Reference Architecture and ETSI Multi-access Edge Computing (MEC) specifications guide the development and deployment of edge technologies \cite{antonini2019fog, giust2017multi}.

Batching is a key technological approach to enhance inference computation efficiency, widely applied in cloud-based AI applications \cite{olston2017tensorflow,nvidia2022triton,moritz2018ray}. Through combining a batch of input tensors of the same size into a higher-dimensional tensor, inference with batching can fully exploit the parallel tensor operation capabilities of accelerators (such as GPUs and TPUs) \cite{nvidia2018v100, jouppi2023tpu, nvidia2020jax}. Batching offers higher computational efficiency than sequentially processing the input data, thereby reducing overall latency and enhancing throughput \cite{olston2017tensorflow,nvidia2022triton}. For instance, batching can enhance server computational efficiency by over 100\% compared to those without batching \cite{inoue2021queueing}. Extensive work has been conducted on batching within cloud computing environments, wherein schemes of presetting static batch sizes are designed \cite{olston2017tensorflow,nvidia2022triton,moritz2018ray}.

As edge servers are rather resource-constrained, computation-efficient batching becomes even more critical when implementing inference applications. However, existing studies on cloud batching often fall short when applied to edge computing, primarily due to the \textit{dynamics} of task arrivals at edge nodes \cite{abbas2017mobile}. Unlike cloud servers, which experience relatively stable and high-density task arrivals, edge nodes face significantly lower and more sporadic task arrivals due to the spatially constrained coverage. Accordingly, edge servers usually wait for a longer time to form computation-efficient batches, compromising the timeliness of edge computing.

In this paper, we present an Adaptive and Cooperative Batching (ACBatch) framework to achieve low-latency edge inference. Specifically, we answer the two following key questions:

 **Question 1.** *How to balance computational efficiency and batch waiting time over dynamic task arrivals?*

An effective strategy involves optimizing the completion time of dynamic task arrival sequences through the adaptive adjustment of batch sizes. Therefore, it is essential to quantify the relationship among batch size, computational efficiency, and batch waiting time. To this end, we measure batch efficiency across diverse computing platforms and popular edge inference neural networks. The results show that batch efficiency increases sub-linearly with batch size. We introduce a general model to describe this relationship quantitatively. Furthermore, we observe a sequential correlation between tasks and batches on batch size optimization and thus propose the Dynamic Programming-based Adaptive Batching (DPAB) algorithm to minimize the completion time of a task sequence with low complexity. DPAB mainly provides a solution to fit the traffic dynamics in the temporal domain. Additionally, the multi-access of edge servers motivates us to rethink the problem in the spatial domain, which leads to the second question.

 **Question 2.** *How to mitigate the influence of traffic dynamics and enhance the overall performance from the network aspect?*

By leveraging the multi-access technologies, a mobile user can be flexibly steered to other surrounding access points and edge servers for service. Thus, we can steer and concentrate the closely arrived tasks at specific edge servers, whereby the waiting time for batching at each edge server can be reduced without compromising computation efficiency. Following this idea, we jointly optimize adaptive batching and cooperative traffic steering to enhance the performance of ACBatch. The formulated problem is proved to be NP-hard, and we propose a heuristic algorithm named Multi-Server Cooperative Batching (MSCB) to solve the problem.

Our main results and key contributions are summarized as follows:

* *Novel Problem Formulation:* We introduce a batch efficiency model to quantify the relationship between computing efficiency and batch size based on the measurement results of diverse computing platforms and neural networks. Then, the ACBatch framework is proposed and optimized to minimize the completion time of a task sequence. To our knowledge, this is the first study on the joint design of adaptive batching and cooperative traffic steering for edge inference.
* *Efficient Batching Scheme Design:* We start from the simplified single-server case and propose the DPAB algorithm, which adaptively optimizes the batch size to fit the dynamic inference task arrivals. Extending DPAB to multiple servers, we incorporate spatial task steering and propose the heuristic iterative algorithm MSCB. MSCB optimizes batching and steering jointly for the general multi-server environment, achieving low complexity and performance guarantee.
* *Performance Verification:* Real-trace experimental results demonstrate an average performance improvement of 89.17\% in completion time and 76.52\% in latency compared to state-of-the-art baselines. Empirical results also show that ACBatch effectively handles edge computing scenarios with extensive workloads, significant burstiness, and high spatial aggregation. In addition, ablation studies reveal that our joint optimization of batching and steering enhances the computational efficiency to 1.24x, compared to an ACBatch variation without steering, while adding the waiting time by 0.11x.

The rest of this paper is organized as follows. Section [[ACBatch/index#II. RELATED WORK|II]] introduces related studies. Section [[ACBatch/index#III. SYSTEM MODEL|III]] describes the system model, and Section [[ACBatch/index#III. SYSTEM MODEL|IV]] formulates the optimization problem. In Section [[ACBatch/index#V. ACBATCH OPTIMIZATION|V]], we present the Dynamic Programming-based Adaptive Batching algorithm for the single-server case. Next, we extend it to the Multi-Server Cooperative Batching algorithm to solve the general multi-server case. Experimental results are shown in Section [[ACBatch/index#VI. PERFORMANCE EVALUATIONS|VI]]. We conclude this paper in Section [[ACBatch/index#VII. CONCLUSION|VII]].

## II. RELATED WORK

## III. SYSTEM MODEL

## IV. PROBLEM FORMULATION AND ANALYSIS

## V. ACBATCH OPTIMIZATION

## VI. PERFORMANCE EVALUATIONS

## VII. CONCLUSION



---

## REFERENCES

\[1\] E. Li, L. Zeng, Z. Zhou, and X. Chen, “Edge ai: On-demand accelerating deep neural network inference via edge computing,” IEEE Transactions on Wireless Communications, vol. 19, no. 1, pp. 447–457, 2019. ^li2019edge