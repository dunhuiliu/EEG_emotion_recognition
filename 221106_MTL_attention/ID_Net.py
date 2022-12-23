import torch.nn as nn
import torch.nn.functional as F

class ID_Net(nn.Module):
    def __init__(self):
        super(ID_Net, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(10368, 50)
        self.fc2 = nn.Linear(50, 31)
        self.fc3 = nn.Linear(31, 31)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10368)
        x = self.fc1(x)
        x = F.relu(x)
        x1 = self.fc2(x)
        x2 = self.fc3(x1)
        return x2, x2
        return x1, x2