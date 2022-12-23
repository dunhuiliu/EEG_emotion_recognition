import torch.nn as nn
import torch.nn.functional as F

class EmoNet(nn.Module):
    def __init__(self):
        super(EmoNet, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
        x = self.fc3(x)
        return x