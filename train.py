import os
from random import shuffle
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models.cnn

device = "cuda" if torch.cuda.is_available() else "cpu"

class RHDDataset(Dataset):
    def __init__(self, data_folder, data_type='training', transform=None, target_transform=None):
        if data_type != 'training' and data_type != 'evaluation':
            raise Exception('Data type could be only "evaluation" or "training".')
        with open(os.path.join(data_folder, data_type, 'anno_%s.pickle' % data_type), 'rb') as file:
            self.image_labels = pickle.load(file)
        self.image_dir = os.path.join(data_folder, data_type, 'color')
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_labels)
    
    def __getitem__(self, idx):
        image = read_image(os.path.join(self.image_dir, '%.5d.png' % idx))
        label = self.image_labels[idx]['xyz']
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def train(dataloader, model, optimizer, criterion, epoch):

    writer = SummaryWriter()
    n_iter = 0
    for ep in tqdm(range(epoch)):
        for image, label in (pbar := tqdm(dataloader)):
            
            n_iter += 1

            image = image.to(device)
            label = label.to(device)

            image = image.type(torch.float)

            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"epoch {ep + 1}/{epoch}; loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), n_iter)


def main():
    dataset = RHDDataset(data_folder='data/RHD_published_v2/', data_type='training')
    dataloader = DataLoader(dataset, batch_size=32)
    learning_rate = 1e-3
    epoch = 5
    criterion = nn.MSELoss()
    model = models.cnn.CNN(320, 320).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(dataloader, model, optimizer, criterion, epoch)
    
    return 0

if __name__ == "__main__":
    main()

        