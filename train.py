import os
import json
from os.path import exists
from transformers import AutoFeatureExtractor
import torch

from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from tqdm import tqdm
from models.layers import Swin
device = "cuda" if torch.cuda.is_available() else "cpu"
writer = SummaryWriter()

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
class MyDataset(Dataset):
    def __init__(self, data_folder, data_type='train', transform=None, target_transform=None):
        self.data_folder = data_folder
        self.data_type = data_type
        self.transform =transform
        self.target_transform = target_transform
        with open(os.path.join(self.data_folder, self.data_type, 'ids.json'), 'r') as jsonfile:
            self.ids = json.load(jsonfile)
        self.size = len(self.ids)
        self.ids = self.ids

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.data_folder, self.data_type, '%s.png' % self.ids[idx]))
        image = feature_extractor(image, return_tensors="pt")['pixel_values']
        image = torch.squeeze(image, dim=0)
        with open(os.path.join(self.data_folder, self.data_type, '%s.json' % self.ids[idx]), 'r') as jsonfile:
            label = json.load(jsonfile)
        label = label['pts3d_2hand']
        label = torch.Tensor(label)
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def test(test_dataset, model, criterion, pbar, batch_size):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    total_loss = 0
    total_size = 0
    for image, label in test_dataloader:
        total_size += 1
        model.eval() 
        image = image.to(device)
        label = label.to(device) # 32 42 3
        image = image.type(torch.float)
        with torch.no_grad():
            pred = model(image)
        loss = criterion(pred, label)
        total_loss += loss
        pbar.set_description(f"calculating test loss: {total_loss/total_size:.8f}; test batch: {total_size}/{int(len(test_dataset) / batch_size + 1)}")
    return total_loss / total_size

def train(train_dataset, test_dataset, model, optimizer, criterion, batch_size, curr_epoch, epoch):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter()

    if not os.path.exists(os.path.join('data', 'ckpts')):
        os.mkdir(os.path.join('data', 'ckpts'))
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_folder = os.path.join('data', 'ckpts', 'ckpt_' + timestamp)
    os.mkdir(checkpoint_folder)

    ckpt_count = 0
    n_iter = 0
    pbar = tqdm(total=int(len(train_dataset) / batch_size + 1))
    for ep in range(curr_epoch, epoch):
        ckpt_count += 1
        pbar.reset()
        for image, label in train_dataloader:
            pbar.update(1)
            model.train()
            n_iter += 1

            image = image.to(device)
            label = label.to(device)

            image = image.type(torch.float)

            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"epoch {ep + 1}/{epoch}; loss: {loss.item():.8f}")
            writer.add_scalar('Loss/train', loss.item(), n_iter)
        test_loss = test(test_dataset, model, criterion, pbar, batch_size)
        writer.add_scalar('Loss/test', test_loss, n_iter)
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss.item(),
            'test_loss': test_loss,
            }, os.path.join(checkpoint_folder, 'ckpt_' + str(ckpt_count) + '.pt'))
    pbar.close()
        


def main(batch_size=32, learning_rate=1e-5, epoch=500, ckpt_path = None):
    train_dataset = MyDataset(data_folder=os.path.join('data', 'hiu_dmtl_data'), data_type='train')
    test_dataset = MyDataset(data_folder=os.path.join('data', 'hiu_dmtl_data'), data_type='test')
    criterion = nn.MSELoss()
    model = Swin().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    curr_epoch = 0
    if ckpt_path != None and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_epoch = checkpoint['epoch']
        #train_loss = checkpoint['train_loss']
        #test_loss = checkpoint['test_loss']
    train(train_dataset, test_dataset, model, optimizer, criterion, batch_size, curr_epoch, epoch)

    

    return 0


if __name__ == "__main__":
    main()
