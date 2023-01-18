from .basic import BasicDataset
from .woslice import WOSliceDataset
from .textinfo import TextInfoDataset
from .wlabel import WlabelDataset
from .woslicewlabel import WOSliceWlabelDataset
from .textinfowlabel import TextInfoWlabelDataset

MODEL_LIST = ["DKT", "SAKT", "DKVMN", "TRKT"]

def get_ds(model, additional):
    assert model in MODEL_LIST
    if model == "DKVMN":
        if additional == True :
            return WOSliceWlabelDataset
        else : 
            return WOSliceDataset
    elif model == "TRKT" :
        if additional == True :
            return TextInfoWlabelDataset
        else : 
            return TextInfoDataset
    elif additional == True :
        return WlabelDataset
    else : 
        return BasicDataset