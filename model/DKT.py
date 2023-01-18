import torch
import torch.nn as nn

class DKT(nn.Module):
    def __init__(self, in_dim,cfg):
        super(DKT, self).__init__()
        self.in_dim = in_dim
        self.emb_dim = cfg.edim
        self.hidden_dim = cfg.hdim
        self.out_dim = cfg.item
        self.dr_Rate = cfg.dr
        self.device = cfg.device
        self.hidden = torch.zeros(1,self.hidden_dim)
        self.cell = torch.zeros(1,self.hidden_dim)
        nn.init.normal_(self.hidden)
        nn.init.normal_(self.cell)
        self.embedding = nn.Linear(in_features = self.in_dim, out_features = self.emb_dim, bias = False)
        self.lstm = nn.LSTM(input_size = self.emb_dim, hidden_size = self.hidden_dim, num_layers = 1, batch_first = True, dropout = self.dr_Rate)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        bat_size = input.shape[0]
        hidden, cell = torch.stack([self.hidden]*bat_size, dim=1).to(self.device), torch.stack([self.cell]*bat_size, dim=1).to(self.device)
        x = self.embedding(input)
        hidden, _ = self.lstm(x,(hidden,cell))
        pred = self.classifier(hidden)
        return self.sigmoid(pred)

    def posterior_predict(self, pred, target_item):
        target_item = torch.unsqueeze(target_item, -1) # Change a tensor into size (bat_size, len, 1) to match the dimension of pred
        posterior_item_predict = torch.gather(input = pred, index = target_item, dim = -1)
        posterior_item_predict = torch.squeeze(posterior_item_predict, dim = -1)
        return posterior_item_predict