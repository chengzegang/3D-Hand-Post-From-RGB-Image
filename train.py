import os
import json
from os.path import exists
from transformers import AutoFeatureExtractor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from tqdm import tqdm
from models.layers import Swin
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def train(dataloader, model, optimizer, criterion, epoch, save_model_param_path=None):
    writer = SummaryWriter()
    n_iter = 0
    for ep in range(epoch):
        for image, label in (pbar := tqdm(dataloader)):
            n_iter += 1

            image = image.to(device)
            label = label.to(device) # 32 42 3

            image = image.type(torch.float)

            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"epoch {ep + 1}/{epoch}; loss: {loss.item():.8f}")
            writer.add_scalar('Loss/train', loss.item(), n_iter)
        if save_model_param_path != None:
            torch.save(model.state_dict(), save_model_param_path)


def main(batch_size=32, learning_rate=1e-6, epoch=12, load_model_param=True, load_model_param_path='./params/param.pt',
         save_model_param_path='./params/param.pt'):
    dataset = MyDataset(data_folder=os.path.join('data', 'hiu_dmtl_data'), data_type='train')
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.MSELoss()
    model = Swin().to(device)
    if load_model_param and exists(load_model_param_path):
        model.load_state_dict(torch.load(load_model_param_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(dataloader, model, optimizer, criterion, epoch, save_model_param_path='./params/param.pt')

    

    return 0


if __name__ == "__main__":
    main()
