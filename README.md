# Awesome-Causality-and-ML-Papers
This is a repository for organizing articles related to causal discovery, invariant learning, and machine learning. Most papers are linked to **my reading notes** and **my slides summaries**.

# Table of Contents (ongoing)
* [Causal discovery](#causal-discovery)
   * [Differentiable causal discovery](#differentiable-causal-discovery)
   * [Differentiable causal discovery from heterogeneous-nonstationary-scale change data](#differentiable-causal-discovery-from-heterogeneous-nonstationary-scale-change-data)
   * [Traditional causal discovery](#traditional-causal-discovery)
   * [Traditional causal discovery from heterogeneous-nonstationary data](#traditional-causal-discovery-from-heterogeneous-nonstationary-data)
* [Invariant learning](#invariant-learning)
* [Basis of causal inference](#basis-of-causal-inference)
* [Machine learning](#machine-learning)

## Causal discovery

### Differentiable causal discovery
1. NIPS 2018 Spotlight [DAGs with NO TEARS: Continuous Optimization for Structure Learning](https://arxiv.org/abs/1803.01422)(将组合无环约束转为光滑等式约束，通过ALM将等式约束优化转为无约束优化求解；但无环性约束难以保证，不能确保输出DAG。基于MSE的评分函数不合理。只考虑了线性). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/NOTEARS.pdf)
2. ICML 2019 [DAG-GNN: DAG Structure Learning with Graph Neural Networks](https://arxiv.org/abs/1904.10098)(提出更适合在当前深度学习平台下实现的非循环约束，可处理线性/非线性、离散+连续、向量值+标量值数据，通过神经网络建模非线性). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/DAG-GNN.pdf)
3. ICLR 2020 [Gradient-Based Neural DAG Learning](https://arxiv.org/abs/1906.02226)(通过NN建模非线性，提出与NN深度相关的加权邻接矩阵表示，给出了最大似然目标函数下的理论保证). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/GraN-DAG.pdf)
4. NIPS 2020 [On the Role of Sparsity and DAG Constraints for Learning Linear DAGs](https://arxiv.org/abs/2006.10201)(在线性高斯情况下，通过最大似然+软DAG约束结合，取代最小二乘+硬DAG约束，优化更易求解，效果更好，并提供了理论保证). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/GOLEM.pdf)
5. AISTATS 2020 [Learning Sparse Nonparametric DAGs](https://arxiv.org/abs/1909.13189)(通过NN建模非线性，提出与NN深度独立的加权邻接矩阵表示). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/NOTEARS-MLP%2BSOB.pdf)
6. ICML 2021 [DAGs with No Curl: An Efficient DAG Structure Learning Approach](https://arxiv.org/abs/2106.07197)(基于图外微积分理论提出DAG新表示，并基于图Hodge理论提出更高效优化算法，隐式确保无环约束，避免多次迭代). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/NoCurl.pdf)
7. NIPS 2022 [DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization](https://arxiv.org/abs/2209.08037)(提出基于对数行列式函数的无环性表示，并提出一种新的优化算法来取代ALM，在线性和非线性情况下，较大提升了求解精度和速度). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/DAGMA.pdf)

### Differentiable causal discovery from heterogeneous-nonstationary-scale change data
1. KDD 2021 [DARING: Differentiable Causal Discovery with Residual Independence](https://dl.acm.org/doi/10.1145/3447548.3467439)(在评分函数中引入残差独立性约束，通过对抗学习优化两套参数，能与任何可微因果发现结合，提升CD准确率，进一步在域注释未知情况下解决异质数据中的CD). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/DARING.pdf)
2. NIPS 2021 [Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game](https://arxiv.org/abs/2102.13647)(本文表明可微因果发现框架性能主要会受到方差可排序性影响，不具备尺度不变性). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/Varsortability.pdf)
3. arXiv 2022 [Differentiable Invariant Causal Discovery](https://arxiv.org/abs/2205.15638)(基于IRM+NOTEARS/非线性NOTEARS，解决异质数据中CD；基于IRM，环境要求高，需域注释，异质性局限于噪声分布变化，难在真实数据集上验证)
4. ICLR 2023 [Boosting Differentiable Causal Discovery via Adaptive Sample Reweighting](https://arxiv.org/abs/2303.03187)(通过样本自适应加权，能与任何可微因果发现结合，提升CD准确率，进一步在域注释未知情况下解决异质数据中的因果发现；性能与CD骨干模型高度相关，异质性局限于噪声方差变化). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/ReScore.pdf)
5. KDD 2023 Oral [Discovering Dynamic Causal Space for DAG Structure Learning](https://arxiv.org/abs/2306.02822)(图结构集成到评分函数中，能更好地刻画估计DAG与真实DAG间距离，进一步解决异质数据中CD；异质性局限于噪声均值变化，框架需基于可微CD). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/CASPER.pdf)
6. arXiv 2023 [Structure Learning with Continuous Optimization: A Sober Look and Beyond](https://arxiv.org/abs/2304.02146)(相比于方差可变排序性，更合理给出了可微框架表现优劣的原因，并展望了后续改进方向). [[Reading Notes]](https://zhuanlan.zhihu.com/p/674880782) 
7. NIPS 2023 [A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models](https://arxiv.org/abs/2303.18211)(基于R^{2}可排序性的尺度不变性，解决线性情况下尺度变化数据中的因果发现). [[Reading Notes]](https://zhuanlan.zhihu.com/p/673527734) 
   

### Traditional causal discovery
1. JMLR 2007 [Estimating high-dimensional directed acyclic graphs with the PC-algorithm](https://arxiv.org/abs/math/0510436)(PC算法通过条件独立性测试来确定因果骨架，并通过定向准则来确定因果方向，最终得到MEC)
2. JMLR 2002 [Optimal structure identification with greedy search](https://dl.acm.org/doi/10.1162/153244303321897717)(GES定义一个评分函数用于衡量MEC与观测数据的拟合程度，并搜索DAG空间以找到得分最高的MEC；GES采用局部启发式方法对DAG进行搜索)
3. JMLR 2006 [A Linear Non-Gaussian Acyclic Model for Causal Discovery](https://dl.acm.org/doi/10.5555/1248547.1248619)(LiNGAM假设变量间的函数关系为线性非高斯，通过原因与残差的独立性判断因果方向，同时将LiNGAM模型转为ICA的形式高效求解)
4. NIPS 2008 [Nonlinear causal discovery with additive noise models](https://arxiv.org/abs/2206.06243)(ANM假设变量间的函数关系为非线性加性噪声模型，通过原因与残差的独立性判断因果方向)
5. UAI 2009 [On the Identifiability of the Post-Nonlinear Causal Model](https://arxiv.org/abs/1205.2599)(PNL假设变量间的函数关系为后非线性模型，通过原因与残差的独立性判断因果方向，并可证明LiNGAM与ANM是PNL的特殊形式)
6. JMLR 2011 [DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model](https://arxiv.org/abs/1101.2489)(DirectLiNGAM相比于LiNGAM求解速度变慢，但精度和收敛性变好，同时尺度稳定)

### Traditional causal discovery from heterogeneous-nonstationary data
1. IJCAI 2015 [Identification of Time-Dependent Causal Model: a gaussian process treatment](https://dl.acm.org/doi/10.5555/2832581.2832745)(通过扩展高斯过程回归，能得因果模型系数如何随时间变化（全时图）)
2. NIPS 2017 [Learning Causal Structures Using Regression Invariance](https://arxiv.org/abs/1705.09644)(利用不同域下回归系数的变化区分因果方向；只适用于线性)
3. NIPS 2018 [Multi-domain causal structure learning in linear systems](https://dl.acm.org/doi/10.5555/3327345.3327524)(利用因果模块间的独立性解决多域因果发现；只适用于线性，大量独立性测试，很耗时)
4. ICML 2019 [Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models](https://arxiv.org/abs/1905.10857)(在非线性状态空间模型的框架下形式化了非平稳环境中的因果发现和预测；只适用于线性，仅因果方向识别)
5. Neural Networks 2020 [FOM: Fourth-order moment based causal direction identification on the heteroscedastic data](https://dl.acm.org/doi/abs/10.1016/j.neunet.2020.01.006)(在异质数据中，提出噪声的四阶矩来衡量因果方向的不对称性；仅因果方向识别，局限于噪声方差变化)
6. JMLR 2020 [Causal Discovery from Heterogeneous/Nonstationary Data with Independent Changes](https://arxiv.org/abs/1903.01672)(能在异质和非平稳数据中CD，能恢复潜变量因果影响+非平稳驱动力，变量可为向量值数据；但需大量独立性测试，很耗时。需域或时间注释的先验。需基于伪因果充分性假设)

## Invariant learning

1. JRSSB 2016 [Causal inference by using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332)(不变因果预测ICP基于多环境训练数据，利用假设检验得到预测目标Y的因果父节点集合；只适用于线性情况且所得因果父节点集是输入特征的子集，不适用于图像和文本等高维复杂数据). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/ICP.pdf)
2. Journal of Causal Inference 2018 [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576)(非线性ICP通过考虑条件独立性测试，突破ICP中线性高斯假设，求得非线性非参数下的父节点集). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/非线性ICP.pdf)
3. JASA 2018 [Invariant Causal Prediction for Sequential Data](https://arxiv.org/abs/1706.08058)(ICP的时序扩展). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/时序ICP.pdf)
4. arXiv 2019 [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)(不变风险最小化IRM基于多环境训练数据，在优化目标中添加跨环境不变约束，学习得到预测目标Y的因果特征，因果特征可以不再是输入特征的子集). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/IRM.pdf)

## Basis of causal inference

1. YouTube 2020 [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course)(POM和SCM框架的基础). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/Basis%20of%20causal%20inference.pdf)
2. The MIT Press 2017 [Elements of causal inference: foundations and learning algorithms](https://library.oapen.org/bitstream/id/056a11be-ce3a-44b9-8987-a6c68fce8d9b/11283.pdf)(书中包括POM和SCM框架，因果发现和时序因果发现基础). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/The%20basis%20of%20causal%20discovery%20from%20time%20series%20data.pdf)

## Machine learning
1. bilibili 2020 [Tensorflow2.0](https://www.bilibili.com/video/BV1B7411L7Qt/?spm_id_from=333.999.0.0&vd_source=618b3a1cb488e3aeed9429a4fc8d9a65)(介绍tensorflow2框架，CNN和RNN)
2. NIPS 2014 [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)(GAN). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/GAN.pdf)
3. arXiv 2013 [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)(VAE). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/VAE.pdf)
4. NIPS 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)(Transformer). [[Slides]](https://github.com/huiyang-yi/Awesome-Causality-and-ML-Papers/blob/main/Slides/Transformer.pdf)



