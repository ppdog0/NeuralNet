# NeuralNet
#### 背景介绍

最初创建这个项目是为了参加微软空间实践站的一个项目[神经网络简明教程-入门项目](https://studentclub.msra.cn/project/institute/22)，由于是入门项目，难度不大，但收获很多。这个教程地址在[这](https://github.com/microsoft/ai-edu/tree/master/B-教学案例与实践/B6-神经网络基本原理简明教程)，有兴趣可以学习。

#### 项目介绍

Linear: 就是这个项目的[作业](https://github.com/microsoft/ai-edu/issues/402)，用于分析两个数据集：波士顿房价数据集(线性回归)和鸢尾花数据集(线性多分类)。代码都是根据教程写的(~~几乎一模一样~~)。写的时候我就发现，这两个数据集库文件有，所有就直接导入了(~~大雾~~)。

NetCode: 就是我学习教程逐步迭代的代码(~~也是一模一样~~)。其实最开始我以为是学习如何编程，到后面逐渐意识到其实最重要的是如何分析以及优化。参数分为两个部分：

- 参数: 权重$w$和偏置$b$
- 超参: 包括训练数据批量大小、学习率、训练次数、隐藏层数和隐藏神经元个数等

优化即是对上述两者进行优化以达到节省资源的目的：

- 参数: 设置初始值; 训练的一部分数据进行保存以便下次加载。
- 超参: 减少训练时间、避免不充分拟合以及过度拟合



目前还在学习深度神经网络，后续还会更新……