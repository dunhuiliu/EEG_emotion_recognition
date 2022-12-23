import torch
import torch.nn as nn
import torch.nn.functional as F

class IndiviTransNet(nn.Module):
    def __init__(self, transformer_phase):
        super(IndiviTransNet, self).__init__()
        self.transformer_phase = transformer_phase
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(10368, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        if self.transformer_phase:
            with torch.no_grad():
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = x.view(-1, 10368)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x1 = F.relu(x)
                x2 = self.fc3(x1)
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(-1, 10368)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x1 = F.relu(x)
            x2 = self.fc3(x1)
        return x1, x2