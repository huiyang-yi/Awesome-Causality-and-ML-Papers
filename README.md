# Awesome-Causality-and-ML-Papers
This is a repository for organizing articles related to causal discovery, invariant learning, and machine learning. Most papers are linked to **my reading notes** and **my powerpoint summaries**.

# Table of Contents (ongoing)
* [Causal discovery](#causaldiscovery)
   * [Differentiable causal discovery](#new)
   * [Differentiable causal discovery from heterogeneous/nonstationary data](#new-heterogeneous/nonstationary)
   * [Traditional causal discovery](#old-but-important)
   * [Traditional causal discovery from heterogeneous/nonstationary data](#old-but-important-heterogeneous/nonstationary)
* [Invariant learning](#invariantlearning)
* [Machine learning](#machinelearning)
  
# Causal discovery

## Differentiable causal discovery
1. NIPS 2018 Spotlight [DAGs with NO TEARS: Continuous Optimization for Structure Learning](https://arxiv.org/abs/1803.01422)(将组合无环约束转为光滑等式约束，通过增广拉格朗日法将等式约束优化转为无约束优化求解；但无环性约束难以保证，不能确保输出DAG。基于MSE的评分函数不合理。只考虑了线性)
2. ICML 2019 [DAG-GNN: DAG Structure Learning with Graph Neural Networks](https://arxiv.org/abs/1904.10098)(提出更适合在当前深度学习平台下实现的非循环约束，可处理线性/非线性、离散+连续、向量值+标量值数据，通过神经网络建模非线性)
3. ICLR 2020 [Gradient-Based Neural DAG Learning](https://arxiv.org/abs/1906.02226)(通过NN建模非线性，提出与NN深度相关的加权邻接矩阵表示，给出了最大似然目标函数下的理论保证)
4. NIPS 2020 [On the Role of Sparsity and DAG Constraints for Learning Linear DAGs](https://arxiv.org/abs/2006.10201)(在线性高斯情况下，通过最大似然+软DAG约束结合，取代最小二乘+硬DAG约束，优化更易求解，效果更好，并提供了理论保证)
5. AISTATS 2020 [Learning Sparse Nonparametric DAGs](https://arxiv.org/abs/1909.13189)(通过NN建模非线性，提出与NN深度独立的加权邻接矩阵表示)
6. ICML 2021 [DAGs with No Curl: An Efficient DAG Structure Learning Approach](https://arxiv.org/abs/2106.07197)(基于图外微积分理论提出DAG新表示，并基于图Hodge理论提出更高效优化算法，隐式确保无环约束，避免多次迭代)
7. NIPS 2022 [DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization](https://arxiv.org/abs/2209.08037)(提出基于对数行列式函数的无环性表示，并提出一种新的优化算法来取代ALM，在线性和非线性情况下，较大提升了求解精度和速度)

## Differentiable causal discovery from heterogeneous/nonstationary data
1. KDD 2021 [DARING: Differentiable Causal Discovery with Residual Independence](https://dl.acm.org/doi/10.1145/3447548.3467439)(在评分函数中引入残差独立性约束，通过对抗学习优化两套参数，能与任何可微因果发现结合，提升CD准确率，进一步在域注释未知情况下解决异质数据中的CD)
2. arXiv 2022 [Differentiable Invariant Causal Discovery](https://arxiv.org/abs/2205.15638)(基于IRM+NOTEARS/非线性NOTEARS，解决异质数据中CD；基于IRM，环境要求高，需域注释，异质性局限于噪声分布变化，难在真实数据集上验证)
3. ICLR 2023 [Boosting Differentiable Causal Discovery via Adaptive Sample Reweighting](https://arxiv.org/abs/2303.03187)(通过样本自适应加权，能与任何可微因果发现结合，提升CD准确率，进一步在域注释未知情况下解决异质数据中的因果发现；性能与CD骨干模型高度相关，异质性局限于噪声方差变化)
4. KDD 2023 Oral [Discovering Dynamic Causal Space for DAG Structure Learning](https://arxiv.org/abs/2306.02822)(图结构集成到评分函数中，能更好地刻画估计DAG与真实DAG间距离，进一步解决异质数据中CD；异质性局限于噪声均值变化，框架需基于可微CD)

## Traditional causal discovery
1. JMLR  [Estimating high-dimensional directed acyclic graphs with the PC-algorithm](https://arxiv.org/abs/math/0510436)(基于独立性约束的因果发现共性：通过条件独立性测试来确定因果骨架，并通过定向准则来确定因果方向，最终得到MEC)
2. NeurIPS [OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling](https://zhuanlan.zhihu.com/p/658191974)(使用在线模型集成，克服时序模型部署过程中遇到的分布变化问题)
3. ICLR [Out-of-Distribution Representation Learning for Time Series Classification](https://arxiv.org/abs/2209.07027)(从OOD的角度考虑时序分类的问题)
4. ICLR [Contrastive Learning for Unsupervised Domain Adaptation of Time Series](https://arxiv.org/abs/2206.06243)(用对比学习对其类间分布为时序DA学一个好的表征)
5. ICLR [Pareto Invarian Risk Minimization](https://openreview.net/forum?id=esFxSb_0pSL)(通过多目标优化角度理解与缓解OOD/DG优化难问题)
6. ICLR [Fairness and Accuracy under Domain Generalization](https://arxiv.org/abs/2301.13323)(不仅考虑泛化的性能，也考虑泛化的公平性)
7. Arxiv [Adversarial Style Augmentation for Domain Generalization](https://arxiv.org/abs/2301.12643)(对抗学习添加图像扰动以提升模型泛化性)
8. Arxiv [CLIPood: Generalizing CLIP to Out-of-Distributions](https://arxiv.org/abs/2302.00864)(使用预训练的CLIP模型，克服domain shift and open class两个问题)
9. SIGKDD [Domain-Specific Risk Minimization for Out-of-Distribution Generalization](https://arxiv.org/abs/2208.08661)(每个域学习单独的分类器，测试阶段根据entropy动态组合)[[Code]](https://github.com/yfzhang114/AdaNPC)[[Reading Notes]](https://zhuanlan.zhihu.com/p/631524930)
10. CVPR [Federated Domain Generalization with Generalization Adjustment](https://scholar.google.com/scholar_url?url=https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf&hl=zh-CN&sa=X&d=13348506996942284912&ei=sTpvZIjhI9OQ6rQP29uDqAU&scisig=AGlGAw8T1YjQNN8nVv2lI6LPBiGS&oi=scholaralrt&hist=lUnt8X4AAAAJ:7797965790415635509:AGlGAw-zJ0qtstLHlwZtiYmf7uNN&html=&pos=1&folt=rel)(为联邦域泛化(FedDG)提供了一个新的新的减小方差的正则项以鼓励公平性)
11. CVPR [Distribution Shift Inversion for Out-of-Distribution Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.pdf)(TTA方法，将OoD测试样本用仅在源分布上训练的扩散模型向训练分布转移然后再测试)
12. CVPR [SFP: Spurious Feature-targeted Pruning for Out-of-Distribution Generalization](https://arxiv.org/abs/2305.11615)(通过移除那些强烈依赖已识别的虚假特征的网络分支来实现modular risk minimization (MRM))
13. CVPR [Improved Test-Time Adaptation for Domain Generalization](https://arxiv.org/abs/2304.04494)(使用一个具有可学习参数的损失函数，而不是预定义的函数)
14. ICLR [Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization](https://openreview.net/forum?id=uyqks-LILZX)(真实世界的数据通常在不同属性上有多种分布偏移，目前DG算法无法work，本文利用数据生成过程的知识自适应地识别和应用正确的正则化约束)
15. ICLR [Using Language to Extend to Unseen Domains](https://openreview.net/forum?id=eR2dG8yjnQ)(利用CLIP模型的知识将源域图像embedding转换为多个目标域的representation（从photos of birds转化为paintings of birds）)
16. ICLR [How robust is unsupervised representation learning to distribution shift?](https://openreview.net/forum?id=LiXDW7CF94J)(无监督学习算法中学习到的表示在各种极端和现实分布变化下的泛化效果优于SL)
17. ICLR [PGrad: Learning Principal Gradients For Domain Generalization](https://openreview.net/forum?id=CgCmwcfgEdH)(测量了所有训练域的训练动态,最终的梯度聚合了并给出一个鲁棒的优化方向，有点像meta-learning)
18. ICLR [Causal Balancing for Domain Generalization](https://openreview.net/forum?id=F91SROvVJ_6)(提出了一种平衡的小批量抽样策略，将有偏差的数据分布转换为平衡分布，基于数据生成过程的潜在因果机制的不变性。)
19. ICLR [Cycle-consistent Masked AutoEncoder for Unsupervised Domain Generalization](https://openreview.net/forum?id=wC98X1qpDBA)(无监督域泛化(UDG)，其中不需要成对的数据来连接不同的域。这个问题的研究相对较少，但在DG背景下是有意义的。)

## Traditional causal discovery from heterogeneous/nonstationary data

0. CVPR Oral [Towards Principled Disentanglement for Domain Generalization](https://zhuanlan.zhihu.com/p/477855079)(将解耦用于DG，新理论，新方法)
1. Arxiv [How robust are pre-trained models to distribution shift?](https://arxiv.org/abs/2206.08871)(自监督模型比有监督以及无监督模型更鲁棒，在小部分OOD数据上重新训练classifier提升很大)
2. ICML [A Closer Look at Smoothness in Domain Adversarial Training](https://arxiv.org/abs/2206.08213)(平滑分类损失可以提高域对抗训练的泛化性能)
3. CVPR [Bayesian Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/528829486)(缓解IRM在模型过拟合时退化为ERM的问题)
4. CVPR [Towards Unsupervised Domain Generalization](https://zhuanlan.zhihu.com/p/528829486)(关注模型预训练的过程对DG任务的影响，设计了一个在DG数据集无监督预训练的算法)
5. CVPR [PCL: Proxy-based Contrastive Learning for Domain Generalization](https://zhuanlan.zhihu.com/p/528829486)(直接采用有监督的对比学习用于DG效果并不好，本文提出可行方法)
6. CVPR [Style Neophile: Constantly Seeking Novel Styles for Domain Generalization](https://zhuanlan.zhihu.com/p/528829486)(本文提出了一种新的方法，能够产生更多风格的数据)
7. Arxiv [WOODS: Benchmarks for Out-of-Distribution Generalization in Time Series Tasks](https://woods-benchmarks.github.io/)(一个关于时序数据OOD的多个benchmark)
8. Arxiv [A Broad Study of Pre-training for Domain Generalization and Adaptation](https://arxiv.org/pdf/2203.11819.pdf)(深入研究了预训练对于DA,DG任务的作用，简单的使用目前最好的backbone足已取得SOTA的效果)
9. Arxiv [Domain Generalization by Mutual-Information Regularization with Pre-trained Models](https://arxiv.org/pdf/2203.10789.pdf)(使用预训练模型的特征指导finetune的过程，提高泛化能力)
10. ICLR Oral [A Fine-Grained Analysis on Distribution Shift](https://zhuanlan.zhihu.com/p/466675818)(如何准确的定义distribution shift，以及如何系统的测量模型的鲁棒性)
11. ICLR Oral [Fine-Tuning Distorts Pretrained Features and Underperforms Out-of-Distribution](https://zhuanlan.zhihu.com/p/466675818)(fine-tuning（微调）和linear probing相辅相成)

# Invariant learning

1. JRSSB [Causal inference by using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332)(不变因果预测ICP基于多环境训练数据，利用假设检验得到预测目标Y的因果父节点集合；只适用于线性情况且所得因果父节点集是输入特征的子集，不适用于图像和文本等高维复杂数据)
2. Journal of Causal Inference [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576)(非线性ICP通过考虑条件独立性测试，突破ICP中线性高斯假设，求得非线性非参数下的父节点集)
3. JASA [Invariant Causal Prediction for Sequential Data](https://arxiv.org/abs/1706.08058)(ICP的时序扩展)
4. arXiv [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)(不变风险最小化IRM基于多环境训练数据，在优化目标中添加跨环境不变约束，学习得到预测目标Y的因果特征，因果特征可以不再是输入特征的子集)
  
# Machine learning

1. JRSSB [Causal inference by using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332)(不变因果预测ICP基于多环境训练数据，利用假设检验得到预测目标Y的因果父节点集合，只适用于线性情况且所得因果父节点集是输入特征的子集，不适用于图像和文本等高维复杂数据)
2. Journal of Causal Inference [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576)(非线性ICP通过考虑条件独立性测试，突破ICP中线性高斯假设，求得非线性非参数下的父节点集)
3. JASA [Invariant Causal Prediction for Sequential Data](https://arxiv.org/abs/1706.08058)(ICP的时序扩展)
4. arXiv [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)(不变风险最小化IRM基于多环境训练数据，在优化目标中添加跨环境不变约束，学习得到预测目标Y的因果特征，因果特征可以不再是输入特征的子集)



