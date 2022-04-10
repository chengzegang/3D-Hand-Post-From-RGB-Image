import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, image_h, image_w):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding='same')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 3, padding='same')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 8, 3, padding='same')
        self.relu4 = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc5 = nn.Linear(8 * image_h * image_w, 42 * 3)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(42 * 3, 42 * 3)
        self.unflat = nn.Unflatten(1, (42, 3))
    
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        x = self.flat(x)
        
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)

        x = self.unflat(x)
        return x