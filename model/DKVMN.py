import torch
import torch.nn as nn

class DKVMN(nn.Module):
    def __init__(self, in_dim, cfg):
        super(DKVMN, self).__init__()
        self.in_dim = in_dim
        self.cnum = cfg.cnum
        self.edim = cfg.edim
        self.inum = cfg.item
        self.device = cfg.device
        self.A = nn.Linear(in_features=self.inum, out_features=self.edim, bias=False)
        self.B = nn.Linear(in_features=self.in_dim, out_features=self.edim, bias=False)
        self.kmat = torch.zeros(self.cnum, self.edim).to(self.device) #fix
        self.vmat = torch.zeros(self.cnum, self.edim).to(self.device) #dynamic
        nn.init.normal_(self.kmat)
        nn.init.normal_(self.vmat)
        self.softmax = nn.Softmax(dim = -1)
        #Read part
        self.linear = nn.Linear(in_features = 2*self.edim, out_features = self.edim)
        self.classifier = nn.Linear(in_features = self.edim, out_features = self.inum)
        #Write part
        self.erase = nn.Linear(in_features = self.edim, out_features = self.edim)
        self.add = nn.Linear(in_features = self.edim, out_features = self.edim)
        #Activation function
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        '''
        input B X 1 X dk
        '''
    def read(self, x, vmat):
        if len(x.shape) == 3 and len(vmat.shape) == 3:
            b = x.shape[0]
        else : 
            print("Need Batch")
            raise ValueError
        bat_kmat = torch.stack([self.kmat]*b,dim=0)
        W = torch.squeeze(torch.matmul(bat_kmat, x))
        W = torch.unsqueeze(self.softmax(W), dim=1)
        r = torch.cat((torch.squeeze(torch.matmul(W, vmat)), torch.squeeze(x)), 1)
        output = self.classifier(self.tanh(self.linear(r)))
        return W, self.sigmoid(output)

    def write(self, x, W, vmat):
        e = self.sigmoid(self.erase(x))
        one = torch.ones(x.shape[0], self.cnum, self.edim).to(self.device)
        we = torch.matmul(torch.transpose(W,-2,-1), torch.unsqueeze(e,dim=1))
        vmat = torch.mul(vmat, one - we)
        a = self.tanh(self.add(x))
        vmat = vmat + torch.matmul(torch.transpose(W,-2,-1), torch.unsqueeze(a,dim=1))
        del one    
        return vmat

    def forward(self, item, interaction):
        if interaction.shape[1] == item.shape[1]:
            b = interaction.shape[0]
            seq = interaction.shape[1]
        else : 
            print("Need Batch")
            raise ValueError
        interaction = self.B(interaction)
        item = self.A(item)
        outs = []
        vmat = torch.stack([self.vmat]*b, dim=0)
        for n in range(seq):
            itr = interaction[:,n,:]
            itm = torch.unsqueeze(item[:,n,:],dim=-1)
            W, output = self.read(itm, vmat)
            outs.append(output)
            vmat = self.write(itr, W, vmat)
        return torch.stack(outs, dim = 1)

    def posterior_predict(self, pred, target_item):
        target_item = torch.unsqueeze(target_item, -1) # Change a tensor into size (bat_size, len, 1) to match the dimension of pred
        posterior_item_predict = torch.gather(input = pred, index = target_item, dim = -1)
        posterior_item_predict = torch.squeeze(posterior_item_predict, dim = -1)
        return posterior_item_predict