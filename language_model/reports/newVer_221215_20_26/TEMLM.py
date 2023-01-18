import math
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, MulticlassF1Score, MulticlassAccuracy

device = torch.device('cpu')


class TEMLM(nn.Module):
    def __init__(self,**TEMLM_kwargs):
        super(TEMLM, self).__init__()
        self.TE = textembedding(**(TEMLM_kwargs['TE_kwargs'])).to(device)
        self.ME = mathembedding(**(TEMLM_kwargs['ME_kwargs'])).to(device)
        self.d_model = TEMLM_kwargs['d_model']
        self.nhead = TEMLM_kwargs['n_head']
        self.num_layers = TEMLM_kwargs['num_layers']
        self.PEdropout = TEMLM_kwargs['PEdropout']
        self.ELdropout = TEMLM_kwargs['ELdropout']
        self.max_len = TEMLM_kwargs['max_len']
        self.mathMask = self.ME.mathMask
        
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, batch_first=True, dropout=self.ELdropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = self.num_layers)
        
        self.PE = PositionalEncoding(self.d_model, self.PEdropout, self.max_len)
        self.Leaky = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.Softmax = nn.Softmax(dim=1)
    
        self.RCCLayer1 = nn.Linear(self.ME.EmbOut,1)
        self.RCCdropout = nn.Dropout(0.1)
        self.MEPLayer = nn.Linear(self.ME.EmbOut, self.ME.emb)
        self.MEPdropout = nn.Dropout(0.1)
        
        self.MSELoss = nn.MSELoss()
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCELoss()
        self.MEPf1 = MulticlassF1Score(num_classes=self.ME.emb,).to(device)
        self.MEPacc = MulticlassAccuracy(num_classes=self.ME.emb,).to(device)
        self.RCCf1 = BinaryF1Score().to(device)
        self.RCCacc = BinaryAccuracy().to(device)

        
    def forward(self, x, requires_grad = True, MEPmode = False):
        """
        input data process mechanism:
        
        text tokenize -> '[MATH]' 있는 곳은 math embedding, 나머지는 textembedding.
        '[MATH]' 토큰은 5로 인코딩 되므로, 5만 detect해서 math embedding을 거친다.
        
        args
            x 
        
        """
        if MEPmode:
            if requires_grad:
                ex, MEPlabel = self.TM_embedding(x[0], x[1], MEPmode=MEPmode)
                # ex = self.PE(ex)
                ex = self.encoder(ex)
                ex = self.tanh(ex)
            else:
                with torch.no_grad():
                    ex, MEPlabel = self.TM_embedding(x[0], x[1], MEPmode=MEPmode)
                    # ex = self.PE(ex)
                    ex = self.encoder(ex)
                    ex = self.tanh(ex)
                
            return ex, MEPlabel
        
        else:
            if requires_grad:
                ex = self.TM_embedding(x[0], x[1], MEPmode=MEPmode)[0]
                # ex = self.PE(ex)
                ex = self.encoder(ex)
                ex = self.tanh(ex)
                
            else:
                with torch.no_grad():
                    ex = self.TM_embedding(x[0], x[1], MEPmode=MEPmode)[0]
                    # ex = self.PE(ex)
                    ex = self.encoder(ex)
                    ex = self.tanh(ex)
            return ex
        
    def TM_embedding(self, x, encEqn, MEPmode = False):

        
        embeddings = []
        maskindexs = []
        if MEPmode == False:
            for i, bat in enumerate(x):
                batEmb = []
                count = 0
                if len(encEqn[i])!=0:
                    eqnEmb = self.ME(encEqn[i])
                for j, h in enumerate(bat):
                    if h==self.TE.MathNum:
                        batEmb.extend(eqnEmb[count])
                        count +=1
                    else:
                        batEmb.append(self.TE(h))
                        
                if len(batEmb)!=0:
                    embeddings.append(torch.stack(batEmb))
                    
                    
            embeddings = self.embeddingAveraging(embeddings)
            return (torch.stack(embeddings),)
            
        else:
            for i, bat in enumerate(x):
                batEmb = []
                count = 0
                maskFind = False
                if len(encEqn[i])!=0:
                    eqnEmb = self.ME(encEqn[i])
                for j, h in enumerate(bat):
                    if h==self.TE.MathNum:
                        if self.mathMask in encEqn[i][count]:
                            maskFind = True
                            maskindex = encEqn[i][count].index(self.mathMask)
                            maskindexs.append(len(batEmb) + maskindex)             
                        batEmb.extend(eqnEmb[count])
                        count +=1
                    else:
                        batEmb.append(self.TE(h))
                if len(batEmb)!=0:
                    embeddings.append(torch.stack(batEmb))
                if maskFind == False:
                    maskindexs.append(0)
                    
            embeddings = self.embeddingAveraging(embeddings)
        
            return (torch.stack(embeddings),maskindexs)
    
    def embeddingAveraging(self, embeddings):
        new_embeddings = []
        totalAvg = int(sum([emb.shape[0] for emb in embeddings])/len(embeddings))+1
        PADtensor = torch.tensor([self.d_model*[self.TE.padNum]], device = device)
        for emb in embeddings:
            if emb.shape[0]<totalAvg:
                new_embeddings.append(torch.concat((emb, PADtensor.repeat(totalAvg-emb.shape[0], 1))))
            else:
                new_embeddings.append(emb[:totalAvg])
        
        return new_embeddings
        
    def embeddingMaximizing(self, embeddings):
        new_embeddings = []
        totalMax = int(max([emb.shape[0] for emb in embeddings]))
        PADtensor = torch.tensor([self.d_model*[self.TE.padNum]], device = device)
        for emb in embeddings:
            if emb.shape[0]<totalMax:
                new_embeddings.append(torch.concat((emb, PADtensor.repeat(totalMax-emb.shape[0], 1))))
        
        return new_embeddings
    
    def MEPLoss(self, MEPpred,MEPlabel=None, maskindexs=None, textenc = None, encEqn = None, CE = True, MSE = False):
        if CE ==True and MSE ==True:
            CE, f1, acc = self.MEPLoss_CE(MEPpred, maskindexs, MEPlabel)
            MSE = self.MEPLoss_MSE(MEPpred, maskindexs, MEPlabel, textenc, encEqn)
            return CE+MSE, f1, acc
        elif CE ==True and MSE == False:
            return self.MEPLoss_CE(MEPpred, maskindexs, MEPlabel)
        elif CE == False and MSE ==True:
            return self.MEPLoss_MSE(MEPpred, maskindexs, MEPlabel, textenc, encEqn), 0, 0
    
    def MEPLoss_CE(self, MEPpred, maskindexs, MEPlabel):
        """
        Masked Equation Prediction
        """
        
        x = self.MEPLayer(MEPpred)    
        # x = self.MEPdropout(x)
        index = self.__MEP_indexMaking(maskindexs, x.shape)
        x = torch.gather(x, 1, index).squeeze(1)
        # x = x[:,0,:].squeeze(1)
        x = self.Softmax(x)
        f1 = self.MEPf1(x, MEPlabel)
        acc = self.MEPacc(x, MEPlabel)
        
        return self.CELoss(x, MEPlabel), f1, acc
    
    #def MEPLoss_MSE(self, MEPpred, maskindexs, MEPlabel, textenc, encEqn):
    #    """
    #    Masked Equation Prediction
    #    """
    #    mep_loss = 0
    #    predLength = MEPpred.shape[1]
    #    encList = self.originEqnBatchlist(encEqn, maskindexs, MEPlabel, predLength)
    #    OriginEqnEmb = self.forward((textenc, encList), requires_grad = False)
    #    # OriginEqnEmb = self.MEPLayer(OriginEqnEmb)
    #    mep_loss += self.MSELoss(OriginEqnEmb, MEPpred)
    #        
    #    if mep_loss!=0:
    #        return mep_loss/len(maskindexs)
    #    else:
    #        return 0
    
    def __MEP_originPred(self, MEPpred, maskindexs, MEPlabel):
        originPred = MEPpred.detach().clone()
        batLen = MEPpred.shape[0]
        MEPlabel_emb = self.ME(MEPlabel, requires_grad = False).detach()
        
        for bat in range(batLen):
            if maskindexs[bat]!=0:
                originPred[bat,maskindexs[bat],:] = MEPlabel_emb[bat]
            
        return originPred
        
    def __MEP_indexMaking(self, indexes, shape):
        for i, ind in enumerate(indexes):
            if ind>=shape[1]:
                indexes[i] = 0
        index = torch.tensor([shape[2]*[ind] for ind in indexes]).unsqueeze(1).to(device)
        return index
        
    def RCCLoss(self, RCCpred, RCClabel):
        """
        Related Context Classification
        
        50%: original input data
        50%: 한 문항의 [MATH]들을 다른 문항들의 [MATH]로 바꿈
        
        original인지 아닌지에 대한 classification error loss
        
        """
        RCClabel = RCClabel.to(device)
        x = self.RCCLayer1(RCCpred[:,0,:].squeeze(1)).squeeze(-1)  
        # x = self.RCCdropout(x)
        # x = x[:,0,:].squeeze(1)
        x = self.sigmoid(x)
        
        f1 = self.RCCf1(x, RCClabel)
        acc = self.RCCacc(x, RCClabel)
        
        return self.BCELoss(x, RCClabel), f1, acc


class textembedding(nn.Module):
    def __init__(self, **textembedding_kwargs):
        super(textembedding, self).__init__()
        self.emb = textembedding_kwargs['emb']
        self.textEmb = textembedding_kwargs['textEmb']
        self.EmbOut = textembedding_kwargs['EmbOut']
        self.padNum = textembedding_kwargs['PAD']
        self.MathNum = textembedding_kwargs['Math']
        self.embedding = nn.Embedding(self.emb, self.textEmb, device = device)
        self.linear = nn.Linear(self.textEmb, self.EmbOut)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        ex = self.embedding(torch.tensor(x, dtype = torch.int64, device=device))
        ex = self.linear(ex)
        
        ex = self.tanh(ex)
        return ex
        
        

class mathembedding(nn.Module):
    def __init__(self, **mathembedding_kwargs):
        super(mathembedding, self).__init__()
        self.emb = mathembedding_kwargs['emb']
        self.mathEmb = mathembedding_kwargs['mathEmb']
        
        self.d_model = mathembedding_kwargs['d_model']
        self.nhead = mathembedding_kwargs['nhead']
        self.num_layers = mathembedding_kwargs['num_layers']
        
        self.dropout = mathembedding_kwargs['dropout']
        self.max_len = mathembedding_kwargs['max_len']
        
        self.EmbOut = mathembedding_kwargs['EmbOut']
        self.mathMask = mathembedding_kwargs['mathMask']
        
        self.tanh = nn.Tanh()
        self.PE = PositionalEncoding(self.d_model, self.dropout, self.max_len)
        self.embedding = nn.Embedding(self.emb, self.mathEmb)
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = self.num_layers)
        
        self.linear = nn.Linear(self.mathEmb, self.EmbOut)
        
    def forward(self, x, requires_grad = True):
        if requires_grad:
            ex = self.embedding(torch.tensor(x, dtype = torch.int64, device = device))
            # ex = self.PE(ex)
            ex = self.encoder(ex)
            ex = self.linear(ex)
            
        else:
            with torch.no_grad():
                ex = self.embedding(torch.tensor(x, dtype = torch.int64, device = device))
                # ex = self.PE(ex)
                ex = self.encoder(ex)
                ex = self.linear(ex)
                
        return self.tanh(ex)
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)