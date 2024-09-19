# DP-DISTILL: fine-tune and distillation kit for pre-trained Deep Potential models
[DP-distill](https://github.com/ruoyuwang1995nya/dp-distill) automates the finetune and distillation process which enables practical atomistic simualtion with the highly transferable, but also computationally expensive DPA-2 pre-trained models. The project is currently built upon the  [DPGEN2](https://github.com/deepmodeling/dpgen2) workflow.

## Table of Contents

- [DP-distill: Workflow for finetune and fast distillation](#voltcraft-battery-simulation-workflow-automation)

  - [1. Overview](#1-overview)
  - [2. Installation](#2-installation)
  - [3. Quick Start](#3-quick-start)
  - [4. User Guide](#4-user-guide)


## 1. Overview
Inspired by DPGEN concurrent learning scheme, DP-distill provides automated workflow for efficient model fine-tune and distillation towards practical application of DPA-2 pretrained model in the field of *atomic simulation*. Fig.1 shows the basic workflow of model finetune. Given the initial structure input of training systems, the workflow generates perturbed structures, and executes a series of short *ab initio* molecular dynamics (AIMD) simulation based upon randomly perturbed structures. The pretrained model is firstly finetuned by the AIMD dataset, then DeePMD simulation with the fine-tuned model would search new configurations, which are then labeled by first-principle calculation softwares such as ABACUS. If the fine-tuned model cannot predict the labeled dataset with sufficient accuracy, the collected dataset would be added to the fine-tune training set, and the *train-search-label* process would iterate until convergence is achieved. 
<div>
    <img src="./docs/images/fine-tune.png" alt="Fig1" style="zoom: 35%;">
    <p style='font-size:1.0rem; font-weight:none'>Figure 1. Fine-tune workflow.</p>
</div>

A lightweight DeePMD model can also be generated from a pre-trained/fine-tuned DPA-2 model through distillation, which enables much faster simulation of given systems. The distilled model can be generated with much less GPU resources and negligible CPU cost compared with standard DP-GEN pipeline which involves simultaneous training of multiple randomly initialized models, provided that a well-converged DPA-2 platform model is availiable. Figure 2 shows the schematic of the distillation workflow.
 <div>
    <img src="./docs/images/distillation.png" alt="Fig2" style="zoom: 35%;">
    <p style='font-size:1.0rem; font-weight:none'>Figure 2. Distillation workflow.</p>
</div>

## 2. Installation
DP-distill can be built and installed form the source.
```shell
git clone https://github.com/ruoyuwang1995nya/dp-distill.git
pip install .
```

## 3. Quick start
DP-distill can be accessed from CLI interface. For instance, a finetune workflow can be submitted by the following command:
```shell
dp-dist submit finetune.json -t finetune
```
The `finetune.json` specifies imput parameters of the finetune task, whose details can be found in the `examples` directory of this repository. 

## 4. Userguide
Examples of json input file for model finetune and distillation can be found in the `examples` directory. 