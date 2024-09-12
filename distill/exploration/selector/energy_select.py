from .conf_selector import ConfSelector
from pathlib import Path
import dpdata
class EnergySelect(ConfSelector):
    def select(
        self,
        system:Path|str,
        test_res:dict,
        type_map=None,
        **config
        ):
        sys=dpdata.System(
            system,
            fmt='deepmd/npy',
            type_map=type_map)
        atom_numb=test_res["atom_numb"]
        numb_frame=sys.get_nframes()
        train_e=test_res["details"]["train_e"]
        pred_e=test_res["details"]["pred_e"]
        delta_e=abs(train_e-pred_e)/atom_numb
        assert numb_frame == len(delta_e)
        delta_e_conv=config.get("delta_e",0.01)
        select_conf_idx=[]
        for ii in range(numb_frame):
            if delta_e > delta_e_conv:
                select_conf_idx.append(ii)
        return select_conf_idx