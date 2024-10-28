# Input script guide

This part gives a concise description for writing the input script of PFD-kit. The input parameter is very similar to that of [DPGEN2](https://github.com/deepmodeling/dpgen2.git), whose online [documantation](https://docs.deepmodeling.com/projects/dpgen2/en/latest/index.html) might also be useful.

## Basics
### Workflow host
PFD-kit is built upon the dflow package, which is in turn based on the Python API of ARGO workflow. Thus, one needs to specify the address of dflow host as well as the storage server for artifacts.

```json
"dflow_config" : {
	"host" : "http://address.of.the.host:port"
    },
"dflow_s3_config" : {
	"endpoint" : "address.of.the.s3.sever:port"
    },
```
This is only neccesary when submitting to a custom server, there is no need to specify workflow server when running on Bohrium platform. Instead, one needs to input their Bohrium account details:

```json
"bohrium_config": {
        "username": "urname",
        "password": "123456",
        "project_id": 123456,
        "_comment": "all"
    },
```
The workflow would be then hosted on `https://workflows.deepmodeling.com` and the workflow progress can be accessed via `https://workflows.deepmodeling.com/workflows`.

### Node setting
The one has to specify the computation resources, i.e., image and machine type for each step as PFD-kit executes computation tasks within each step via a docker container managed by the workflow host.  

The `default_step_config` define the default step setting, for example:

```json
"default_step_config": {
        "template_config": {
            "image": "registry.dp.tech/dptech/deepmd-kit:2024Q1-d23cf3e"
        },
        "executor": {
            "type": "dispatcher",
            "image_pull_policy": "IfNotPresent",
            "machine_dict": {
                "batch_type": "Bohrium",
                "context_type": "Bohrium",
                "remote_profile": {
                    "input_data": {
                        "job_type": "container",
                        "platform": "ali",
                        "scass_type": "c2_m4_cpu"}
                }
            }
        }
    },
```
The `image` entry in the `template_config` defines the image on which the container would be generated. The `executor` entry defines the machine and relevant parameters. In this case, a container node with two CPU core and 4GB memory would be instantiated via the Bohrium platform.

Similary, to set the step performing, say, the DFT calculation with ABACUS software, one can add the `run_fp_config` setting in the `step_configs` entry:   

```json
"step_configs": {
        "run_fp_config": {
            "template_config": {
                "image": "registry.dp.tech/dptech/abacus:3.6.1",
            },
            "continue_on_success_ratio": 0.9,
            "executor": {
                "type": "dispatcher",
                "image_pull_policy": "IfNotPresent",
                "machine_dict": {
                    "batch_type": "Bohrium",
                    "context_type": "Bohrium",
                    "remote_profile": {
                        "input_data": {
                            "job_type": "container",
                            "platform": "ali",
                            "scass_type": "c32_m64_cpu"}}}}}
                            }
```

One can see that an ABACUS image  and a more powerful machine with 32 cores and 64GB memories are selected.  

## Fine-tune task
A model fine-tune workflow is defined by the following paramters. One first specifies the workflow type in the `task` section,
which is `finetune` in this case. 

```json
"task":{
        "type":"finetune"
    }
```
The `inputs` section includes essential user input parameters. Here one needs to specify `type_map`, which maps the type embedding of pretrained model 
to corresponding element name.

```json
"inputs":{
    "type_map":["Li","..."]
}
```
The `conf_generation` section defines how perturbed structures are generated from initial structure files. Multiple input structure files can be 
present in the `init_configuration` sub-section, and the `pert_generation` defines the rule for generating perturbed structures.
```json
"conf_generation":{
    "init_configurations":{
        "type": "file",
        "fmt": "vasp/poscar",
        "files": ["LGPS.vasp"]
        },
    "pert_generation":[{
        "conf_idx": "default",
        "atom_pert_distance":0.15,
        "cell_pert_fraction":0.03,
        "pert_num": 5}]
}
```
The `exploration` section defines the 
```json
"exploration": {
        "max_iter":2,
        "convergence":{
            "type":"energy_rmse",
            "RMSE":0.01},
        "filter":[{
            "type":"distance"
        }],
        "type": "lmp",
        "config": {
            "command": "lmp -var restart 0",
            "shuffle_models": false, 
            "head": null},
        "stages":[[
            {  "_comment": "group 1 stage 1 of finetune-exploration",
                "conf_idx": [0],
                "n_sample":3,
                "exploration":{
                "type": "lmp-md",
                "ensemble": "npt",
                "dt":0.005,
                "nsteps": 2000,
                "temps": [500],
                "press":[1],
                "trj_freq": 200},
                "max_sample": 10000
                    }]]},

    "fp": {
        "type": "fpop_abacus",
        "task_max": 50,
        "extra_output_files:":[],
        "run_config": {
            "command": "OMP_NUM_THREADS=4 mpirun -np 8 abacus | tee log"
        },
        "inputs_config": {
            "input_file": "INPUT.scf",
            "pp_files": {
                "Li": "./pp/Li_ONCV_PBE-1.0.upf",
                "S":"./pp/S_ONCV_PBE-1.0.upf",
                "Ge": "./pp/Ge_ONCV_PBE-1.0.upf",
                "P": "./pp/P_ONCV_PBE-1.0.upf"
            },
            "orb_files":{
                "Li": "./orb/Li_gga_9au_100Ry_6s2p.orb",
                "S":"./orb/S_gga_9au_100Ry_3s3p2d.orb",
                "Ge": "./orb/Ge_gga_9au_100Ry_3s3p3d2f.orb",
                "P": "./orb/P_gga_9au_100Ry_3s3p2d.orb"
            }
        },
        "_comment": "fp parameters for calculation"
    },
    "train": {
        "comment":"Training script for downstream DeePMD model",
        "type": "dp",
        "config": {
            "impl": "pytorch",
            "init_model_policy": "no",
            "init_model_with_finetune":true,
            "_comment": "all"
        },
        "template_script": "train.json",
        "init_models_paths":["OpenLAM_2.1.0_27heads_2024Q1.pt"],
        "numb_models":1,
        "_comment": "the initial pre-trained model at 'init_models_paths'"
    }
```
The workflow type is specified in the `task` entry. The `inputs` entry defines key user inputs. 
```json
"inputs":{
    "type_map":["Li","..."]
}
```

### `conf_generation`





## Distillation task
The setting for the distillation tasks are very similar. 


# Arguments of input script