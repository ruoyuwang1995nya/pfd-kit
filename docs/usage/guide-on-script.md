# Input Script Guide

<style>
  p {
    text-align: justify;
  }
</style>

## Basics
### Host and Nodes
#### Workflow Host
PFD-kit is built on the `dflow` package, which uses the Python API of `ARGO` workflows. While designed for cloud-based workflows with `Kubernetes`, a local "debug" mode is available for convenience. No cloud services are required for local execution.

To submit workflows to a remote server, specify the following configuration:

```json
"dflow_config": {
    "host": "http://address.of.the.host:port"
},
"dflow_s3_config": {
    "endpoint": "address.of.the.s3.server:port"
},
```
For the `Bohrium` platform, use:

```json
"bohrium_config": {
    "username": "your_username",
    "password": "your_password",
    "project_id": 123456,
    "_comment": "all"
},
```
The workflow will be hosted on `https://workflows.deepmodeling.com`, and progress can be tracked at `https://workflows.deepmodeling.com/workflows`.

#### Node Settings
The `step_configs` section defines computational resources for tasks like DFT calculations, model training, and MD exploration. For local execution, no configuration is needed, as all tasks run on the local machine.

For remote HPC nodes (e.g., `Slurm` systems), configure as follows:

```json
"step_configs": {
    "run_fp_config": {
        "template_config": {},
        "executor": {
            "type": "dispatcher",
            "host": "your_host",
            "username": "your_username",
            "password": "your_password",
            "port": 22,
            "private_key_file": null,
            "remote_root": "/remote_root",
            "queue_name": "queue",
            "machine_dict": {
                "remote_profile": {
                    "timeout": 600
                }
            },
            "resources_dict": {
                "source_list": ["path_to_source_file"],
                "module_list": ["remote_module"],
                "custom_flags": ["custom_commands"]
            }
        },
        "template_slice_config": {
            "group_size": 1,
            "pool_size": 1
        }
    }
}
```
For `Kubernetes` services (e.g., `Bohrium`), specify the container image and machine type:

```json
"step_configs": {
    "run_fp_config": {
        "template_config": {
            "image": "vasp_image_paths"
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
                        "scass_type": "c32_m64_cpu"
                    }
                }
            }
        }
    }
}
```

## Fine-Tuning
The example `si_ft.json` file defines workflow tasks. Specify the task type (e.g., "finetune") and skip initial data generation and training if the `Domains_SemiCond` branch already provides sufficient accuracy:

```json
"task": {
    "type": "finetune",
    "max_iter": 5,
    "init_ft": false,
    "init_train": false
}
```
The `inputs` section includes essential parameters and model files. Prepare exploration systems with proper perturbation in advance:

```json
"inputs": {
    "base_model_path": "DPA2_medium_28_10M_beta4.pt",
    "init_confs": {
        "prefix": "./",
        "confs_paths": ["./pert_si32.extxyz"]
    },
    "init_fp_confs": {
        "prefix": "./",
        "confs_paths": []
    }
}
```
The `exploration` section defines structural configuration exploration. In this example, MD simulations at 1000 K under varying pressures generate new configurations:

```json
"exploration": {
    "type": "ase",
    "config": {
        "calculator": "dp",
        "head": "Domains_SemiCond"
    },
    "stages": [
        [
            {
                "conf_idx": [0],
                "n_sample": 1,
                "ens": "npt",
                "dt": 2,
                "nsteps": 2000,
                "temps": [1000],
                "press": [1, 1000, 10000],
                "trj_freq": 10
            }
        ]
    ]
}
```
The `select_confs` node filters unphysical configurations and compresses data using entropy-based measures:

```json
"select_confs": {
    "max_sel": 60,
    "frame_filter": [
        {"type": "distance"}
    ],
    "h_filter": {
        "chunk_size": 5,
        "_comment": "entropy-based filter"
    }
}
```
The `fp` section defines DFT calculation settings, including VASP input and pseudopotential files:

```json
"fp": {
    "type": "vasp",
    "task_max": 50,
    "run_config": {
        "command": "mpirun -n 32 vasp_std"
    },
    "inputs_config": {
        "incar": "INCAR.fp",
        "pp_files": {
            "Si": "POTCAR"
        },
        "kspacing": 0.2
    }
}
```
The `train` section specifies the pretrained model and training configuration:

```json
"train": {
    "type": "dp",
    "config": {
        "impl": "pytorch",
        "head": "Domains_SemiCond"
    },
    "template_script": "train.json"
}
```
The `evaluate` section tests the model against a test dataset. Iterations continue until convergence or the maximum cycle limit is reached. Convergence is achieved when the force RMSE falls below 0.06 eV/Å:

```json
"evaluate": {
    "test_size": 0.3,
    "model": "dp",
    "head": "Domains_SemiCond",
    "_comment": "The percentage for test",
    "converge": {
        "type": "force_rmse",
        "RMSE": 0.06
    }
}
```

## Distillation
The distillation script is similar to fine-tuning, with key differences. Change the `task/type` to `dist` and label new frames using the fine-tuned model:

```json
"task": {
    "type": "dist",
    "max_iter": 5
},
"inputs": {
    "base_model_path": "path_to_teacher_model",
    ...
},
"train": {
    "type": "dp",
    "config": {
        "impl": "pytorch"
    },
    "template_script": "./dist_train.json"
},
"fp": {
    "type": "ase",
    "run_config": {
        "model_style": "dp",
        "inputs_config": {
            "batch_size": 500
        }
    }
}
```
