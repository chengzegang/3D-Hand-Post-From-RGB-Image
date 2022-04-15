import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, image_h, image_w):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding='same')
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)
        self.norm4 = nn.BatchNorm2d(128)

        self.flat = nn.Flatten()
        self.fc5 = nn.Linear(51200, 42 * 3)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(42 * 3, 42 * 3)
        self.unflat = nn.Unflatten(1, (42, 3))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = self.relu3(self.conv3(x))
        x = self.pool3(x)
        x = self.norm3(x)

        x = self.relu4(self.conv4(x))
        x = self.pool4(x)
        x = self.norm4(x)

        x = self.flat(x)
        # print(x.shape)
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)

        x = self.unflat(x)
        return x
