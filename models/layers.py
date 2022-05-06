from transformers import AutoFeatureExtractor, SwinModel
import torch.nn as nn
import torch

class Swin(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.linear1 = nn.Linear(768, 768)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(768, 768)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(768, 3)
    

    def forward(self, x):
        x = self.swin(x)
        x = x.last_hidden_state
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.linear3(x)
        x = x[:, :42]
        return x
