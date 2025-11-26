# Quickstart
<style>
  p {
    text-align: justify;
  }
</style>
This section outlines the basic steps to get started with PFD-kit.

## Installation
Install PFD-kit directly from the source using `pip`:
```shell
pip install git+https://github.com/ruoyuwang1995nya/pfd-kit.git
```

## Job Submission
PFD-kit provides a simple CLI interface. For example, submit a fine-tuning workflow with:
```shell
pfd submit finetune.json
```
The `finetune.json` file specifies input parameters for the fine-tuning task. Example files are available in the `examples` directory.

PFD-kit is built on [dflow](https://github.com/dptech-corp/dflow.git) and utilizes OPs from [DPGEN2](https://github.com/deepmodeling/dpgen2.git). Besides local deployment, the cloud-based [Bohrium](https://bohrium.dp.tech) platform is also recommended.

## Tutorial: Diamond Silicon
This tutorial demonstrates the general workflow of PFD-kit using a simple example of crystalline Si in a diamond structure. Relevant input files are located in `/examples/silicon`. In this example, we aim to generate an efficient deep potential model of Si with high accuracy for large-scale atomic simulations. This can be achieved through fine-tuning and distillation using PFD-kit.

### Generate Initial Structures
Generate initial structures for exploration by perturbing a single frame of diamond Si at `si.poscar`. Ten perturbed frames with 32 atoms each will be generated at `pert_si.extxyz` using the following command:
```bash
pfd perturb -n 10 si.poscar -r 2 2 1
```

### Fine-Tuning
This example demonstrates fine-tuning using the DPA-2 pretrained model ([DPA-2.3.1-v3.0.0rc0](https://www.aissquare.com/models/detail?pageType=models&name=DPA-2.3.1-v3.0.0rc0&id=287)) with multiple prediction heads. The `Domains_SemiCond` head, trained on DFT calculations of semiconductor materials, is used to fine-tune a VASP Si model. Prepare the DPA-2 model file, its training script (downloadable from [AIS square](https://www.aissquare.com/models/detail?pageType=models&name=DPA-2.3.1-v3.0.0rc0&id=287)), and the pseudopotential file for Si. The directory structure for fine-tuning is as follows:
```bash
examples/
├── c_Si/ 
│   ├── finetune 
│   │   ├── DPA2_medium_28_10M_rc0.pt 
│   │   ├── ft_*.json 
│   │   ├── input_torch_medium.json 
│   │   ├── INCAR.fp 
│   │   ├── POSCAR  
│   │   └── POTCAR
```

#### Job Submission
Modify one of the input script files starting with `ft_*.json`. This JSON file defines various aspects of the PFD workflow. For this tutorial, we recommend using a script that submits VASP remote jobs via the `Slurm` scheduler while performing other tasks locally (requires `deepmd-kit` installed locally). Alternatively, you can run the workflow entirely on a local machine or on the Bohrium cloud platform. Specify your login credentials to submit VASP jobs:
```json
"step_configs": {
    "run_fp_config": {
    "template_config": {},
    "executor": {
        "type": "dispatcher",
        "host": "your host",
        "username": "your username",
        "password": "your password",
        "port": 22,
        "private_key_file": null,
        "remote_root": "/remote_root",
        "queue_name":"queue",    
        "machine_dict": {
                    "remote_profile": {
                        "timeout": 600
                        }},
        "resources_dict":{
            "source_list":["path_to_source_file"],
            "module_list": ["remote_module"],
            "custom_flags": ["custom_commands"]
                }},
        "template_slice_config": {
                "group_size": 1,
                "pool_size": 1
            }
        }
}
```
For Bohrium or custom Kubernetes services, configure the host information and specify computing resources, including image name and machine types. Examples for running PFD workflows on Bohrium are provided.

Run the workflow with the following command:
```bash
export DFLOW_MODE='debug'&&export DFLOW_DEBUG_COPY_METHOD='copy'&&pfd submit ft_local_slurm_executor.json
```
> For Bohrium, use `pfd submit ft_bohr.json`

A workflow ID will be printed to the console. Check the workflow progression with the `status` subcommand:
```bash
$ pfd status si_ft.json WORKFLOW_ID
+-------------+------------+------------+--------------+---------------+------------------+-------------+
|   iteration | type       |   criteria |   force_rmse |   energy_rmse |   selected_frame | converged   |
+=============+============+============+==============+===============+==================+=============+
|         000 | Force_RMSE |       0.06 |    0.0501339 |  0.00093      |               60 | True        |
+-------------+------------+------------+--------------+---------------+------------------+-------------+
```  
The fine-tuning task completes in one iteration after collecting 60 frames. Download the final model to `~/results/model/task.0000/model.ckpt.pt` using the `download` subcommand:
```bash
pfd download ft_local_slurm_executor.json WORKFLOW_ID
```
The fine-tuned model achieves high accuracy, with an energy prediction error of 0.002 eV/atom and a force prediction error of 0.056 eV/Å.

<div style="text-align: center;">
    <img src="../_static/ft_test.png" alt="Fig2" style="zoom: 100%;">
    <p style='font-size:1.0rem; text-align: center; font-weight:none'>Figure 2. Prediction error of the fine-tuned Si model.</p>
</div>

### Distillation
The fine-tuned model retains the network structure of the pretrained model, which may be inefficient for large-scale simulations. Knowledge distillation transfers the fine-tuned model's knowledge to a lightweight Deep Potential model with a local descriptor by training it on synthetic data generated from the fine-tuned model.

The distillation workflow is similar to fine-tuning, except DFT calculations are replaced by the fine-tuned model, and the output model is trained from scratch. Follow the same procedure to generate initial structures using the `perturb` subcommand. The input script is also similar, with minor differences. Since DFT calculations are unnecessary, the workflow can run locally:
```bash
export DFLOW_MODE='debug'&&pfd submit dist_local.json
```
Compared to fine-tuning, the number of labeled frames per iteration increases to 1500, with 300 used as a test dataset. The distillation ends after one iteration, and the "student" model achieves similar accuracy to the fine-tuned model but runs much faster:
<div style="text-align: center;">
    <img src="../_static/dist_test.png" alt="Fig3" style="zoom: 100%;">
    <p style='font-size:1.0rem; text-align: center;font-weight:none'>Figure 3. Prediction error of the Si model generated through knowledge distillation.</p>
</div>

<div style="text-align: center;">
    <img src="../_static/efficiency.png" alt="Fig4" style="zoom: 50%;">
    <p style='font-size:1.0rem; text-align: center;font-weight:none'>Figure 4. Comparison between the inference efficiency of the PF and PFD model.</p>
</div>