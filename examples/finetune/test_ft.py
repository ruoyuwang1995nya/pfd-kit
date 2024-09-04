import sys
import json
from dflow import Workflow
sys.path.append('../../')
from distill.entrypoint.submit import (
    submit_ft
)
from distill.entrypoint.args import normalize as normalize_args
from distill.entrypoint.common import (
    global_config_workflow,
    expand_idx
)
from distill.entrypoint.submit import workflow_finetune

def reuse_step(wf_id=None):
    if wf_id:
        pass
    else:
        return []
    old_wf=Workflow(id=wf_id)
    step_ls=[]
    
    step_ls.append(old_wf.query_step(key="finetune--pert-gen")[0])
    step_ls.append(old_wf.query_step(key="iter-000--md-expl")[0])
    step_ls.append(old_wf.query_step(key="iter-000--collect-data")[0])
    step_ls.append(old_wf.query_step(key="iter-000--prep-run-fp")[0])
    step_ls.append(old_wf.query_step(key="iter-000--collect-data")[0])
    step_ls.append(old_wf.query_step(key="iter-000--validation-test")[0])
    step_ls.append(old_wf.query_step(key="iter-001--md-expl")[0])
    #step_ls.append(old_wf.query_step(key="iter-001--collect-data")[0])
    step_ls.append(old_wf.query_step(key="iter-001--prep-run-fp")[0])
    #step_ls.append(old_wf.query_step(key="iter--001")[0])
    #step_ls.append(old_wf.query_step(key="001--prep-run-train")[0])
    #step_ls.append(old_wf.query_step(key="001--prep-run-lmp")[0])
    #step_ls.append(old_wf.query_step(key="001--prep-run-fp")[0])
    #step_ls.append(old_wf.query_step(key="001--collect-data")[0])

    #step_ls.extend(old_wf.query_step(key="fastop--run-fp-group"))
    print(step_ls)
    return step_ls 

if __name__ == '__main__':
    with open('ft.json', 'r') as f:
        config_dict = json.load(f)
        
    wf_config = normalize_args(config_dict)
    global_config_workflow(wf_config)
    ft_step=workflow_finetune(wf_config)
    
    wf = Workflow(
        name="aimd-true"#wf_config["name"],
        #parallelism=wf_config["parallelism"]
        )
    wf.add(ft_step)
    #if not no_submission:
    #    wf.submit(reuse_step=reuse_step)
    #return wf    
    
    wf.submit(
    #reuse_step=reuse_step(wf_id="aimd-true-ad1c9")
    )
    #wf=submit_ft(
    #    wf_config=config_dict,
        #reuse_step=reuse_step(wf_id="based-bxcpt")
       # no_submission=True
    #)
    #print(reuse_step(wf_id=""))
    