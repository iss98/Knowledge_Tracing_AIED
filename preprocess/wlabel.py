import torch
from tqdm import tqdm
from .base import BaseDataset
import json

class WlabelDataset(BaseDataset):
    def __init__(self, cfg):
        super(WlabelDataset, self).__init__(cfg)
        self.change_item_name()
        self.input_size = 2 * self.max_item + 23
    
    def change_item_name(self):
      with open('preprocess/prob_data_label.json', 'r', encoding = 'utf-8') as f:
        prob_dict = json.load(f)
      for t in tqdm(self.datalist):
        temp = []
        for item in t["item"]:
            temp.append(prob_dict[item]["label"])
        t["item"].replace(self.item_dict.keys(), self.item_dict.values(), inplace=True)
        t["item"] = torch.tensor(t["item"].values)
        t["label"] = temp

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        temp_dict = self.datalist[idx]
        interaction = temp_dict["item"] + self.max_item * temp_dict["response"]
        interaction = self.one_hot(interaction[:-1], 2 * self.max_item)
        interaction = torch.cat((interaction, torch.tensor(temp_dict["label"][:-1])),dim=-1)
        one_hot_item = self.one_hot(temp_dict["item"][1:], self.max_item)
        return dict(
                input_interaction = interaction,
                input_item = one_hot_item,
                target_item = temp_dict["item"][1:],
                target_response = temp_dict["response"][1:]
            )