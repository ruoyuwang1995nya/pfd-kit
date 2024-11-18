# Overview

[PFD-kit](https://github.com/ruoyuwang1995nya/dp-distill) automates the fine-tuning and distillation process of pre-trained atomic models, such as [DPA-2](https://github.com/deepmodeling/deepmd-kit.git). It enables practical atomistic simualtion with the highly transferable, but computationally expensive pre-trained models. PFD-kit is built upon the  [DPGEN2](https://github.com/deepmodeling/dpgen2) workflow and supports the Deep Potential models. The best experience of PFD-kit is found on Bohrium platform.

## Backgrounds (Only Chinese ver. for now)
结合了量子力学原理与数据驱动算法的机器学习力场模型(e.g. Deep Potential)，可以在量子力学精度下实现超大尺度原子模拟，突破了传统方法无法兼顾计算尺度和精度的困境。然而目前机器学习力场模型普遍存在泛化能力不足、训练集构建成本过高、训练需要经验丰富专家等问题，以上问题导致机器学习势训练所需的时间与计算成本较高，人工参与的内容较多。因此在一些特定问题上，机器学习力场训练采样会存在本质困难。如图1所示，分别代表了构型空间分布广以及元素空间分布广的两种材料体系：1)碳材料。如果是简单的体相碳，只需要采样少数的金刚石结构。又由于金刚石结构具有非常高的空间对称性，因此计算DFT所需的原胞大小也是非常小的，模型训练的总成本非常低。但是一旦要训练碳材料的表面结构，就不得不考虑其表面的多种多样的复杂结构，这些结构往往不是一个原胞尺寸就能描述的，需要构建超大的原胞才能描述。另外由于表面结构的复杂性，需要采样远比简单块体材料数目多的构型，这导致模型总的训练成本极具增加。同理，无序碳材料的模型训练也会存在成本激增的问题；2)对于元素空间分布广的体系，例如高熵合金。元素在空间中的不同排布，所带来的元素之间相互作用也不尽相同。当元素数目增多起来，空间中的元素排列组合可能性也会随之指数爆炸增长，想要将它们全部采样到是几乎不可能的。

<div>
    <img src="../_static/difficulut_case_MLFF.png" alt="Fig1" style="zoom: 35%;">
    <p style='font-size:1.0rem; font-weight:none'>Figure 1. Hard case for machine learning force field.</p>
</div>

通用大原子模型(large atom model, LAM)通过在大量相互关联的数据集上进行预训练，凝练不同应用问题中化学知识和原子构型知识的共通信息，有望解决上述问题。预训练的通用大原子模型具有一定的泛化能力，但对具体领域的描述不够准确，仍需进行模型微调。针对特定领域仅需少量数据集进行模型微调就可以得到高精度专用微调模型，微调训练数据需求量通常是从头训练机器学习力场模型的十分之一到百分之一。微调模型能以远低于DFT的计算成本以及DFT的精度预测材料微观结构的能量与原子受力，从而替代DFT进行高效的机器学习力场数据采样。采样大量的数据后进行模型蒸馏即可构建能够精确描述特定领域并且参数数量较少的蒸馏模型，可应用于数百万原子和纳秒尺度的动力学模拟。具体可应用的场景如二维材料、超材料、半导体材料、新能源材料、合金材料等。(总流程如图2所示) PFD-kit实现了在云端自动化完成从预训练模型(P)到模型微调(F)到模型蒸馏的(D)的全链条过程，相比于从头训练机器学习力场，时间以及计算开销均低一个量级，并且无需人类进行监督指导，彻底解放了模型训练的繁琐流程，为高通量计算和复杂材料体系模拟提供了有效支持。

<div>
    <img src="../_static/workflow.png" alt="Fig2" style="zoom: 35%;">
    <p style='font-size:1.0rem; font-weight:none'>Figure 12. Workflow schematic of PFD-kit.</p>
</div>

## Workflows 
### Fine-tune
Fig.1 shows the schematic of fine-tuning workflow. Given the initial structures of fine-tuning systems, the workflow generates perturbed structures, and executes a series of short *ab initio* molecular dynamics (AIMD) simulation based upon randomly perturbed structures. The pre-trained model is firstly fine-tuned by the AIMD dataset, then MD simulation with the fine-tuned model searches new configurations, which are then labeled by first-principle calculation softwares. If the fine-tuned model cannot predict the labeled dataset with sufficient accuracy, the collected dataset would be added to the fine-tuning training set, and the *train-search-label* process would iterate until convergence. 
<div>
    <img src="../_static/fine-tune.png" alt="Fig3" style="zoom: 35%;">
    <p style='font-size:1.0rem; font-weight:none'>Figure 3. Fine-tune workflow.</p>
</div>

### Distillation
A lightweight model can be generated from a fine-tuned model through distillation, which enables much faster simulation. The distilled model can be generated with training data labeled by the fine-tuned model. Figure 2 shows the schematic of the distillation workflow.
 <div>
    <img src="../_static/distillation.png" alt="Fig4" style="zoom: 25%;">
    <p style='font-size:1.0rem; font-weight:none'>Figure 4. Distillation workflow.</p>
</div>
