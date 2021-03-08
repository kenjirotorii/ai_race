import os
import sys
import cv2
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from autoencoder import Autoencoder


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path, crop=True):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    if crop:
        h = im_rgb.shape[0]
        im_rgb = im_rgb[round(h/2):, :, :]
    return im_rgb


class AEDataSet(Dataset):
    def __init__(self, path_dict, transdorm=None, crop=True):
        """DataSet for Autoencoder

        Args:
            path_dict (dict): [description]
                {
                    0: {
                        "spring": "xxx.jpg",
                        "summer": "xxx.jpg",
                        "autumn": "xxx.jpg",
                        "winter": "xxx.jpg",
                        "normal": "xxx.jpg"
                    },
                    ...
                }
            transdorm ([type], optional): [description]. Defaults to None.
        """
        self.path_dict = path_dict
        self.transform = transdorm
        self.input_imgs = ["spring", "summer", "autumn", "winter"]
        self.target_img = "normal"

    def __len__(self):
        return len(self.path_dict)

    def __getitem__(self, idx):
        input_paths = [self.path_dict[idx][img] for img in self.input_imgs]
        inputs = [get_img(path, crop) for path in input_paths]
        target = get_img(self.path_dict[idx][self.target_img], crop)

        if self.transform:
          inputs = [self.transform(img) for img in inputs]
          target = self.transform(target)

        return inputs, target


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):

    model.train()
    final_loss = 0

    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        for img in inputs:
            img = img.to(device)
            outputs = model(img)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss += loss.item()
        
        scheduler.step()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):

    model.eval()
    final_loss = 0
    
    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        for img in inputs:
            img = img.to(device)
            outputs = model(img)
            loss = loss_fn(outputs, targets)

            final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def inference_and_save(model, path_dict, device, save_path):

    model.eval()

    for k in ["spring", "summer", "autumn", "winter"]:
        img_path = path_dict[k]
        img = get_img(img_path)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        
        outputs = model(img).detach().cpu().squeeze().numpy()

        im_bgr = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        path = save_path + "/" + img_path.split('.')[0] + "_pred_" + k + ".jpg"
        cv2.imwrite(path, im_bgr)

    img_path = path_dict["normal"]
    path = save_path + "/" + img_path
    original = get_img(img_path)
    im_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im_bgr)


def run_training(seed, train_dict, valid_dict):
    
    seed_everything(seed)
    
    train_dataset = AEDataSet(train_dict, transdorm=transforms.ToTensor())
    valid_dataset = AEDataSet(valid_dict, transdorm=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Autoencoder(h=120, w=320, outputs=64)
    
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
    
    loss_fn = nn.MSELoss()
        
    best_loss = np.inf
    
    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        valid_loss = valid_fn(model, loss_fn, validloader, DEVICE)
        print(f"EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            torch.save(model.state_dict(), "autoencoder.pth")


if __name__ == "__main__":

    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5
    EPOCHS = 25