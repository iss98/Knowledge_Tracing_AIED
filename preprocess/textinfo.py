import torch
from tqdm import tqdm
from .base import BaseDataset
import json

from language_model import lm, textTokenizer, mathTokenizer, trainInput
from language_model.ModelInput_old import ModelInput

class TextInfoDataset(BaseDataset):
    def __init__(self, cfg):
        super(TextInfoDataset, self).__init__(cfg)
        self.change_item_name()
        self.input_size = 400 #text embedding 200
    
    def change_item_name(self):
        with open('preprocess/prob_data_label.json', 'r', encoding = 'utf-8') as f:
            prob_dict = json.load(f)
        prob_text_dict = {}
        for item in tqdm(self.item_dict.keys()):
            temp_text = prob_dict[item]["text"]
            temp_eqn = prob_dict[item]["eqn"]
            encodedText = textTokenizer.encode(temp_text)
            encodedEquation = [mathTokenizer.encode(e) for e in temp_eqn]
            inputDict = {'encText': encodedText, 'encEqn':encodedEquation}
            inputDict['unit'] = 1 
            temp_input = trainInput.UC_collate_fn([inputDict])
            temp_output = lm(temp_input)
            emb = temp_output[:,0,:].squeeze()
            emb = torch.tensor(emb.detach().cpu().numpy(), dtype = torch.float32)
            prob_text_dict[item] = emb.unsqueeze(dim=0)
        for t in tqdm(self.datalist):
            temp = []
            for item in t["item"]:
                temp.append(prob_text_dict[item])
            t["item"].replace(self.item_dict.keys(), self.item_dict.values(), inplace=True)
            t["item"] = torch.tensor(t["item"].values)
            t["text_emb"] = temp

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        temp_dict = self.datalist[idx]
        temp_response = torch.cat([temp_dict["response"][:-1].unsqueeze(dim = -1)] * 200, dim = -1)
        text_emb = torch.cat(temp_dict["text_emb"][:-1], dim = 0)
        target_text_emb = torch.cat(temp_dict["text_emb"][1:], dim = 0)
        inter_emb = torch.cat((text_emb, temp_response), dim = -1)
        return dict(
                input_interaction = inter_emb,
                input_item = target_text_emb,
                input_text_emb = text_emb,
                target_item = temp_dict["item"][1:],
                target_response = temp_dict["response"][1:]
            )