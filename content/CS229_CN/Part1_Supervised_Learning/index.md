---
title: 第 I 部分 监督学习
---
<div style="
        height: 80vh;
        display: flex;
        justify-content: center;
        align-items: center;
        background: rgba(0, 0, 0, 0.0);
    ">
        <div style="
            font-size: 36px;
            text-align: center;
        ">
            第 I 部分<br>监督学习
        </div>
</div>

不妨先从几个监督学习的例子谈起。假设有一个数据集，记录有俄勒冈州波特兰市的 $47$ 套房屋的居住面积和价格。

| 居住面积 (平方英尺) | 价格 (千美元) |
| :-----------------: | :-----------: |
|       $2104$        |     $400$     |
|       $1600$        |     $330$     |
|       $2400$        |     $369$     |
|       $1416$        |     $232$     |
|       $3000$        |     $540$     |
|      $\vdots$       |   $\vdots$    |

将这些数据绘制出来：

![[CS229_CN/Part1_Supervised_Learning/figs/house_dataset.svg|500]]

有了这些数据之后，该怎样根据波特兰其他房屋的居住面积来预测其价格呢？

为了后续使用的方便，在这里做以下约定。约定用 $x^{(i)}$ 表示“输入”变量 (在现在的示例中是居住面积)，也称作 **特征 (features)**；用 $y^{(i)}$ 表示要预测的“输出”，或称之为 **目标 (target)** 变量 (这里是房屋价格)。一对 $(x^{(i)}, y^{(i)})$ 称为一个 **训练样本 (training example)**，而用于学习的数据集——由 $n$ 个训练样本组成的列表 $\{(x^{(i)}, y^{(i)}); i = 1, \cdots, n\}$——则称为 **训练集 (training set)**。注意，此处的上标“$i$”仅表示训练集中的索引，而不表示指数运算。此外，用 $\mathcal{X}$ 表示输入的取值空间，$\mathcal{Y}$ 表示输出的取值空间。在示例中，有 $\mathcal{X}=\mathcal{Y}=\mathbb{R}$.

监督学习问题可以形式化地表述为：给定一个训练集，目标是学习一个函数 $h: \mathcal{X} \mapsto \mathcal{Y}$, 该函数的输出 $h(x)$ 可以“很好地”作为相应的 $y$ 的预测。基于历史原因，函数 $h$ 被称为 **假设 (hypothesis)**。整个过程如下图所示：

![[CS229_CN/Part1_Supervised_Learning/figs/learning_process.svg]]

当预测的目标是连续值时 (例如预测房价)，称这类学习问题为 **回归 (regression)** 问题。当 $y$ 只能取有限个离散值时 (例如根据居住面积预测住宅是房屋还是公寓)，则称为 **分类 (classification)** 问题。

| [[CS229_CN/index\|介绍]] | [[CS229_CN/index#目录\|目录]] | ╠ [[chapter1_linear_regression\|第一章]]<br>╠ [[chapter2_classification_and_logistic_regression\|第二章]]<br>╠ [[chapter3_generalized_linear_model\|第三章]]<br>╠ [[chapter4_generative_learning_algorithms\|第四章]]<br>╠ [[chapter5_kernel_methods\|第五章]]<br>╚ [[chapter6_support_vector_machines\|第六章]] |
| :--------------------: | :-----------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
