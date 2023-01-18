import numpy as np
import torch
from torch.utils.data import Dataset

class ModelInput(Dataset):
    def __init__(self, textTokenizer, mathTokenizer, prob_data, unitDict = None):
        
        self.textTokenizer = textTokenizer
        self.mathTokenizer = mathTokenizer
        self.textvocabLength = textTokenizer.vocab_size
        self.mathvocabLenght = mathTokenizer.vocab_size
        self.probNumDict = dict()
        self.prob_data = prob_data

        self.textPad = textTokenizer.encode('[PAD]')[1]
        self.mathPad = mathTokenizer.encode('[PAD]')[1]
        self.mathMask = mathTokenizer.encode('[MASK]')[1]
        self.mathToken = textTokenizer.encode('[MATH]')[1]
        self.eqnnumcheck()
        self.probNumbering()
        if unitDict == None:
            self.unitNumCount()
        else:
            self.unitDict = unitDict
        
    def __getitem__(self, idx):
        text = self.prob_data[self.probNumDict[idx]]['text']
        eqn = self.prob_data[self.probNumDict[idx]]['eqn']
        return self.UCInput(idx, text, eqn)
    
    def __len__(self):
        return self.probNum
    
    
    def eqnnumcheck(self):
        temp_dict = {}
        for key, val in self.prob_data.items():
            mathnum = val['text'].count('[MATH]')
            if len(val['eqn']) >= mathnum:
                temp_dict[key] = val
        self.prob_data = temp_dict
                
    def probNumbering(self):
        for i, key in enumerate(list(self.prob_data.keys())):
            self.probNumDict[i] = key
        self.probNum = len(self.probNumDict)    
        
    
    def unitExtract(self, prob_identifier):
        unit_infoList = prob_identifier.split('-')
        
        course_namedict = {'수학(상)':0, '수학(하)':1, '수학1': 2, '수학2':3, '확률과 통계': 4, '미적분': 5, '기하':6}
        ### 교과명(수1, 수상, 확통 등..)이 있는 데이터 분류 체계인지 확인
        if '[' in unit_infoList[0] and ']' in unit_infoList[0]:
            BigUnitLoc = 1
        else:
            BigUnitLoc = 0
        
        BigUnit = int(unit_infoList[BigUnitLoc][1])-1
        MedUnit = ord(unit_infoList[BigUnitLoc+1][1].lower())-97 # note that ord('a')=97
        prob_num = int(prob_identifier[-1])-1
        
        return self.unitDict[(BigUnit, MedUnit)]
    
    def unitNumCount(self):
        self.unitDict = {}
        unitList = set()
        for key in self.prob_data.keys():
            unit_infoList = key.split('-')
            if '[' in unit_infoList[0] and ']' in unit_infoList[0]:
                BigUnitLoc = 1
            else:
                BigUnitLoc = 0
            
            BigUnit = int(unit_infoList[BigUnitLoc][1])-1
            MedUnit = ord(unit_infoList[BigUnitLoc+1][1].lower())-97 # note that ord('a')=97
            
            unitList.add((BigUnit, MedUnit))
        
        unitList = list(unitList)
        unitList.sort(key = lambda x: (x[0],x[1]))
        
        for i, u in enumerate(unitList):
            self.unitDict[u] = i
            
        self.unitNum = len(unitList)
        
    def unitProbCount(self):
        unitProbNumDict = {i:0 for i in range(self.unitNum)}
        for key, val in self.prob_data.items():
            unitNum = self.unitExtract(key)
            unitProbNumDict[unitNum] +=1
            
        return unitProbNumDict
    ### functions for input generation
    
    def NormalInput(self, text, eqn):
        
        encText = self.textTokenizer.encode(text)
        mathLoc = {i:'' for i in range(len(encText)) if encText[i] == self.mathToken}
        mathnum = len(mathLoc)
        
        encEqn = [self.mathTokenizer.encode(eq) for eq in eqn][:mathnum]
        
        return {"encText":encText, "encEqn":encEqn}
    
    def UCInput(self, idx, text, eqn):
        normal = self.NormalInput(text, eqn)
        normal['unit'] = self.unitExtract(self.probNumDict[idx])
        return normal
   
    
    ### functions for torch.utils.data.DataLoader(collate_fn)
    
    def UC_collate_fn(self, batch):
        batchLength = len(batch)
        texts = [b['encText'] for b in batch]
        encEqns = [b['encEqn'] for b in batch]
        units = [b['unit'] for b in batch]
        
        textbatAvg = int(sum([len(b) for b in texts])/batchLength)
        for i, bat in enumerate(texts):
            if len(bat)<textbatAvg:
                texts[i].extend([self.textPad for _ in range(textbatAvg-len(bat))])
            else:
                texts[i] = bat[:textbatAvg]                    
        
        encEqns = self.encEqnbatAvg(encEqns,None, mode = 'Normal')

        return texts, encEqns, torch.tensor(units)
    
    
    def encEqnbatAvg(self, encEqns, labels, mode):
        
        """
        labels means 'original [MATH]' token in MEP mode.
        In RCC mode, labels means a collection of '0 or 1'
        
        """
        batchLength = len(encEqns)
        avg = 0
        for encEqn in encEqns:
            if len(encEqn)==0:
                continue
            avg += sum([len(enc) for enc in encEqn])/len(encEqn)
        avg =  int(avg/batchLength)+1
        
        for k, encEqn in enumerate(encEqns):
            for i, enc in enumerate(encEqn):
                if len(enc)<avg:
                    encEqns[k][i].extend([self.mathPad for _ in range(avg-len(enc))])
                else:
                    encEqns[k][i] = enc[:avg]
        
        if mode == 'MEP':
            for key, mathloc in enumerate(labels):
                for key, val in mathloc.items():
                    if len(val)<avg:
                        mathloc[key].extend([self.mathPad for _ in range(avg-len(val))])
                    else:
                        mathloc[key] = val[:avg]
                        
        if mode=='Normal':
            return encEqns
            
        return encEqns, labels
    