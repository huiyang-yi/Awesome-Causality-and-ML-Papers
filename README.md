# Awesome-Causality-and-ML-Papers
This is a repository for organizing articles related to causal discovery, invariant learning, and machine learning. Most papers are linked to **my reading notes** and **my powerpoint summaries**.

# Table of Contents (ongoing)
* [Causal discovery](#causaldiscovery)
   * [2023](#2023)
   * [2022](#2022)
   * [2017-2021](#old-but-important)
* [Invariant learning](#invariantlearning)
* [Machine learning](#machinelearning)
# Causal discovery

## 2023
0. ICLR [Free Lunch for Domain Adversarial Training: Environment Label Smoothing](https://arxiv.org/abs/2302.00194)(环境标签平滑，一行代码提升对抗学习的稳定性和泛化性). [[Code]](https://github.com/yfzhang114/Environment-Label-Smoothing)  [[Reading Notes]](https://zhuanlan.zhihu.com/p/600466715)
1. ICML  [AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation](https://arxiv.org/abs/2304.12566)(用KNN进行测试时间自适应，从理论上分析了TTA work的原因)[[Code]](https://github.com/yfzhang114/AdaNPC)  [[Reading Notes]](https://zhuanlan.zhihu.com/p/624770864)
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

## 2022

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

1. JRSSB [Causal inference by using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332)(作者描述了他们早期进行手动红队测试的努力，旨在提高模型的安全性并测量模型的安全性)
2. Journal of Causal Inference [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576)(调查为什么这些越狱攻击成功以及它们如何生成的。竞争目标和不匹配的泛化)
3. JASA [Invariant Causal Prediction for Sequential Data](https://arxiv.org/abs/1706.08058)(通过AI指导来生开发一个有帮助、诚实、无害且不会规避问题的AI助手)
4. arXiv [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)(AUTO-J，相比于传统的评估score，这是一个开源模型，能够有效地评估LLMs在各种任务上的表现。)

# Machine learning

1. JRSSB [Causal inference by using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332)(作者描述了他们早期进行手动红队测试的努力，旨在提高模型的安全性并测量模型的安全性)
2. Journal of Causal Inference [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576)(调查为什么这些越狱攻击成功以及它们如何生成的。竞争目标和不匹配的泛化)
3. JASA [Invariant Causal Prediction for Sequential Data](https://arxiv.org/abs/1706.08058)(通过AI指导来生开发一个有帮助、诚实、无害且不会规避问题的AI助手)
4. arXiv [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)(AUTO-J，相比于传统的评估score，这是一个开源模型，能够有效地评估LLMs在各种任务上的表现。)

