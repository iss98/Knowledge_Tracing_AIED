import torch
from tqdm import tqdm
from .base import BaseDataset

class WOSliceDataset(BaseDataset):
    def __init__(self, cfg):
        super(WOSliceDataset, self).__init__(cfg)
        self.change_item_name()
        self.input_size = 2 * self.max_item 
    
    def change_item_name(self):
      for t in tqdm(self.datalist):
        t["item"].replace(self.item_dict.keys(), self.item_dict.values(), inplace=True)
        t["item"] = torch.tensor(t["item"].values)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        temp_dict = self.datalist[idx]
        interaction = temp_dict["item"] + self.max_item * temp_dict["response"]
        interaction = self.one_hot(interaction, 2 * self.max_item)
        one_hot_item = self.one_hot(temp_dict["item"], self.max_item)
        return dict(
                input_interaction = interaction,
                input_item = one_hot_item,
                target_item = temp_dict["item"],
                target_response = temp_dict["response"]
            )