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
> > 

