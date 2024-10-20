from .force_conv import ForceConvRMSE, ForceConvIdvRMSE
from .energy_conv import EnerConvRMSE

ConvTypes = {
    "force_rmse": ForceConvRMSE,
    "force_rmse_idv": ForceConvIdvRMSE,
    "energy_rmse": EnerConvRMSE,
}
