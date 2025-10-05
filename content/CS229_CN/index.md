---
title: CS229 机器学习 课程讲义
---
作者：Andrew Ng and Tengyu Ma；最近更新：2023 年 6 月 11 日 [(PDF)](https://cs229.stanford.edu/main_notes.pdf)

译者：[Namoe](https://github.com/na-moe)；更新于：2025 年 9 月

## 目录

> [!example]- [[CS229_CN/Part1_Supervised_Learning/index|第 I 部分 监督学习]]  
>   
> > [!example]- [[chapter1_linear_regression|第 1 章 线性回归]]  
> > 
> > ║ ╠ [[chapter1_linear_regression#1.1 最小均方算法|1.1 最小均方算法]]  
> > ║ ╠ [[chapter1_linear_regression#1.2 正规方程|1.2 正规方程]]  
> > ║ ║ ╠ [[chapter1_linear_regression#1.2.1 矩阵导数|1.2.1 矩阵导数]]  
> > ║ ║ ╚ [[chapter1_linear_regression#1.2.2 再探最小二乘法|1.2.2 再探最小二乘法]]  
> > ║ ╠ [[chapter1_linear_regression#1.3 概率解释|1.3 概率解释]]  
> > ║ ╚ [[chapter1_linear_regression#1.4 局部加权线性回归 (选读)|1.4 局部加权线性回归 (选读)]]  
>   
> > [!example]-  [[chapter2_classification_and_logistic_regression|第 2 章 分类与逻辑回归]]  
> > 
> > ║ ╠ [[chapter2_classification_and_logistic_regression#2.1 逻辑回归|2.1 逻辑回归]]  
> > ║ ╠ [[chapter2_classification_and_logistic_regression#2.2 离题：感知机学习算法|2.2 离题：感知机学习算法]]  
> > ║ ╠ [[chapter2_classification_and_logistic_regression#2.3 多类别分类|2.3 多类别分类]]  
> > ║ ╚ [[chapter2_classification_and_logistic_regression#2.4 最大化 ℓ(θ) 的另一种算法|2.4 最大化ℓ(θ) 的另一种算法]]  
>   
> > [!example]-  [[chapter3_generalized_linear_model|第 3 章 广义线性模型]]  
> > 
> > ║ ╠ [[chapter3_generalized_linear_model#3.1 指数族|3.1 指数族]]  
> > ║ ╚ [[chapter3_generalized_linear_model#3.2 构造广义线性模型|3.2 构造广义线性模型]]  
> > ║     &emsp;&nbsp;╠ [[chapter3_generalized_linear_model#3.2.1 普通最小二乘|3.2.1 普通最小二乘]]  
> > ║     &emsp;&nbsp;╚ [[chapter3_generalized_linear_model#3.2.2 逻辑回归|3.2.2 逻辑回归]]  
>   
> > [!example]-  [[chapter4_generative_learning_algorithms|第 4 章 生成式学习算法]]  
> > 
> > ║ ╠ [[chapter4_generative_learning_algorithms#4.1 高斯判别分析|4.1 高斯判别分析]]  
> > ║ ║ ╠ [[chapter4_generative_learning_algorithms#4.1.1 多元正态分布|4.1.1 多元正态分布]]  
> > ║ ║ ╠ [[chapter4_generative_learning_algorithms#4.1.2 高斯判别分析模型|4.1.2 高斯判别分析模型]]  
> > ║ ║ ╚ [[chapter4_generative_learning_algorithms#4.1.3 讨论：GDA 与逻辑回归|4.1.3 讨论：GDA 与逻辑回归]]  
> > ║ ╚ [[chapter4_generative_learning_algorithms#4.2 朴素贝叶斯 (选读)|4.2 朴素贝叶斯 (选读)]]  
> > ║  &emsp;&nbsp;╠ [[chapter4_generative_learning_algorithms#4.2.1 拉普拉斯平滑|4.2.1 拉普拉斯平滑]]  
> > ║  &emsp;&nbsp;╚ [[chapter4_generative_learning_algorithms#4.2.2 文本分类的事件模型|4.2.2 文本分类的事件模型]]  
>   
> > [!example]-  [[chapter5_kernel_methods|第 5 章 核方法]]  
> > 
> > ║ ╠ [[chapter5_kernel_methods#5.1 特征映射|5.1 特征映射]]  
> > ║ ╠ [[chapter5_kernel_methods#5.2 特征的最小均方|5.2 特征的最小均方]]  
> > ║ ╠ [[chapter5_kernel_methods#5.3 使用核技巧的最小均方|5.3 使用核技巧的最小均方]]  
> > ║ ╚ [[chapter5_kernel_methods#5.4 核的性质|5.4 核的性质]]  
>   
> > [!example]-  [[chapter6_support_vector_machines|第 6 章 支持向量机]]  
> > 
> >   &emsp;╠ [[chapter6_support_vector_machines#6.1 间隔：直觉|6.1 间隔：直觉]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.2 符号 (选读)|6.2 符号 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.3 函数间隔与几何间隔 (选读)|6.3 函数间隔与几何间隔 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.4 最优间隔分类器 (选读)|6.4 最优间隔分类器 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.5 拉格朗日对偶 (选读)|6.5 拉格朗日对偶 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.6 最优间隔分类器：对偶形式 (选读)|6.6 最优间隔分类器：对偶形式 (选读)]]  
> >   &emsp;╠ [[chapter6_support_vector_machines#6.7 正则化与非线性可分情况 (选读)|6.7 正则化与非线性可分情况 (选读)]]  
> >   &emsp;╚ [[chapter6_support_vector_machines#6.8 SMO 算法 (选读)|6.8 SMO 算法 (选读)]]  

> [!example]- [[CS229_CN/Part2_Deep_Learning/index|第 II 部分 深度学习]]  
>   
> > [!example]- [[chapter7_deep_learning|第 7 章 深度学习]]  
> > 
> >   &emsp;╠ [[chapter7_deep_learning#7.1 使用非线性模型的监督学习|7.1 使用非线性模型的监督学习]]  
> >   &emsp;╠ [[chapter7_deep_learning#7.2 神经网络|7.2 神经网络]]  
> >   &emsp;╠ [[chapter7_deep_learning#7.3 现代神经网络的模块|7.3 现代神经网络的模块]]  
> >   &emsp;╠ [[chapter7_deep_learning#7.4 反向传播|7.4 反向传播]]  
> >   &emsp;║ ╠ [[chapter7_deep_learning#7.4.1 偏导数初步|7.4.1 偏导数初步]]  
> >   &emsp;║ ╠ [[chapter7_deep_learning#7.4.2 反向传播的通用策略|7.4.2 反向传播的通用策略]]  
> >   &emsp;║ ╠ [[chapter7_deep_learning#7.4.3 基本模块的反向函数|7.4.3 基本模块的反向函数]]  
> >   &emsp;║ ╚ [[chapter7_deep_learning#7.4.4 MLP 的反向传播|7.4.4 MLP的反向传播]]  
> >   &emsp;╚ [[chapter7_deep_learning#7.5 训练样本的向量化|7.5 训练样本的向量化]]  

> [!example]- [[CS229_CN/Part3_generalization_and_regularization/index|第 III 部分 泛化与正则化]]  
>   
> > [!example]- [[chapter8_generalization|第 8 章 泛化]]  
> > 
> >   &emsp;╠ [[chapter8_generalization#8.1 偏差-方差均衡|8.1偏差方差均衡]]  
> >   &emsp;║ ╚ [[chapter8_generalization#8.1.1 (对于回归问题的) 数学分解|8.1.1 (对于回归问题的) 数学分解]]  
> >   &emsp;╠ [[chapter8_generalization#8.2 双下降现象|8.2 双下降现象]]  
> >   &emsp;╚ [[chapter8_generalization#8.3 样本复杂度边界 (选读)|8.3 样本复杂度边界 (选读)]]  
> >   &emsp;&emsp; ╠ [[chapter8_generalization#8.3.1 预备知识|8.3.1 预备知识]]  
> >   &emsp;&emsp; ╠ [[chapter8_generalization#8.3.2 有限 𝓗 的情况|8.3.2 有限 𝓗 的情况]]  
> >   &emsp;&emsp; ╚ [[chapter8_generalization#8.3.3 无限 𝓗 的情况|8.3.2 无限 𝓗 的情况]]  
