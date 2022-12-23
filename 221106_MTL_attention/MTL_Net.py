import torch
import torch.nn as nn
import torch.nn.functional as F
from IndiviTransNet import IndiviTransNet
from EmoNet import EmoNet
from ID_Net import ID_Net
import numpy as np

model_save_path = 'model/transformer_model/'

class MTL_Net(nn.Module):
    def __init__(self, transfomer_group):
        super(MTL_Net, self).__init__()
        
        # 定义身份识别网络
        self.id_net = ID_Net()
        
        # 定义transformer网络
        self.indivNets = transfomer_group
        
        # 定义情感识别网络
        self.emo_net = EmoNet()
        
    def forward(self, x1, x2):
        
        w_out, id_out = self.id_net(x1)
        w = F.softmax(w_out, dim=1)
        
        indivNets_output = []
        for indiv_block in self.indivNets:
            indivNets_output.append(indiv_block(x2)[0].numpy())
        indivNets_output = torch.tensor(np.array(indivNets_output))
        indivNets_output = indivNets_output.transpose(1, 0)
        w = w.unsqueeze(-1)
        w = w.repeat([1, 1, indivNets_output.shape[-1]])
        transformered_feature = (w * indivNets_output).sum(dim=1)
        emo_out = self.emo_net(transformered_feature)
        
        return emo_out, id_out