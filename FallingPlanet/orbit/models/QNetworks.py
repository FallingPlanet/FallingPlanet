import torch
import torch.nn as nn
import torch.nn.functional as F

class DCQN(nn.Module):
    def __init__(self,n_observation ,n_actions):
        super(DCQN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_observation,out_channels=1028,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(1028,512)
        self.conv3 = nn.Conv2d(512,256)
        self.fc = nn.Linear(256,128)
        self.fc = nn.Linear(128,n_actions)

    def forward(self):
        x = F.relu(self.conv1(x))


class DTQN(nn.Module):
    def __init__(self, n_observation, n_actions):
        super(DTQN,self).__init__()
        self.transformer1 = nn.TransformerEncoderLayer(d_model=n_observation)

                 
