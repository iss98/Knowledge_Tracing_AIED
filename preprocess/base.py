import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, cfg):
        super(BaseDataset, self).__init__()
        self.datapath = f"data/result_{cfg.dscor} cor_{cfg.dsitem} item_{cfg.dstime} time.csv"
        self.datalist = []
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.datapath)
        itemlist = []
        for i in tqdm(range(0,len(self.data.index),3)):
            temp_dict = {}
            temp_dict["time_diff"] = torch.tensor(self.data.iloc[i,:].astype('float32').values)
            temp_dict["item"] = self.data.iloc[i+1,:]
            itemlist.extend(list(self.data.iloc[i+1,:]))
            itemlist = list(set(itemlist))
            temp_dict["response"] = torch.tensor(self.data.iloc[i+2,:].astype('float32').values)
            self.datalist.append(temp_dict)
            del temp_dict
        self.item_dict = self.make_dict(itemlist)
        self.max_item = len(self.item_dict.keys())
        del itemlist
        print("Data loading is completed successfully")
                
    def make_dict(self, itemlist):
        item_dict = {}
        for i, item in enumerate(itemlist):
            item_dict[item] = i
        return item_dict

    def one_hot(self, tensor, max_len):
        return F.one_hot(tensor, num_classes = max_len)

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError