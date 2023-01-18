from .DKT import DKT
from .SAKT import SAKT
from .DKVMN import DKVMN
from .TRKT import TRKT

MODEL_LIST = ["DKT", "SAKT", "DKVMN", "TRKT"]

def get_model(name):
    assert name in MODEL_LIST
    if name == "DKT":
        return DKT
    elif name == "SAKT" :
        return SAKT
    elif name == "DKVMN" :
        return DKVMN
    elif name == "TRKT":
        return TRKT
