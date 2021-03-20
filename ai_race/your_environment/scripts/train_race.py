'''
Train vechile control with camera images
'''
import os
import cv2
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from autoencoder import VAE, ControlHead


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
        im_rgb = im_rgb[int(h/2):, :, :]
        
    return im_rgb


class ControlDataSet(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        label = self.img_df.iat[idx, 2]
        img_path = self.img_df.iat[idx, 1]

        img = get_img(img_path, crop=True)

        if self.transform:
            img = self.transform(img.copy())

        return img, label


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):

    model.train()
    final_loss = 0

    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        inputs = inputs.to(device)
        y_pred = model(inputs)
        loss = loss_fn(y_pred, targets)

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
        inputs = inputs.to(device)

        y_pred = model(inputs)
        loss = loss_fn(y_pred, targets)

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def run_training(seed, train_df, valid_df, args):

    seed_everything(seed)
    
    train_dataset = ControlDataSet(train_df, transdorm=transforms.ToTensor())
    valid_dataset = ControlDataSet(valid_df, transdorm=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_z = 128

    model = VAE(h=120, w=320, outputs=num_z)
    model.load_state_dict(torch.load(args.pretrained_model))

    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name == "encoder":
            param.requires_grad = False

    model.decoder = ControlHead(num_z, 3)
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=args.n_epoch, steps_per_epoch=len(trainloader))
        
    best_loss = np.inf
    best_model_path = args.model_path + args.model_name + "_best.pth"

    for epoch in range(args.n_epoch):
        
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        valid_loss = valid_fn(model, loss_fn, validloader, DEVICE)
        print("EPOCH: {}, train_loss: {}, valid_loss: {}".format(epoch, train_loss, valid_loss))
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
        
        if epoch % args.save_model_interval == 4:
            model_path = args.model_path + args.model_name + "_ckpt_{}.pth".format(epoch + 1)
            torch.save(model.state_dict(), model_path)


def parse_args():

    home = os.environ['HOME']
    cwd = os.path.dirname(os.path.abspath(__file__))

    arg_parser = argparse.ArgumentParser(description="Image Classification")

    arg_parser.add_argument("--dataset_name", type=str, default='sim_race')
    arg_parser.add_argument("--data_csv", type=str, default=home+'/Images_from_rosbag/_2020-11-05-01-45-29_2/_2020-11-05-01-45-29.csv')
    arg_parser.add_argument("--model_name", type=str, default='control_model')
    arg_parser.add_argument("--model_path", type=str, default=cwd+'/models/')
    arg_parser.add_argument("--pretrained_model", type=str, default=cwd+'/models/vae_best.pth')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--n_epoch', default=20, type=int, help='The number of epoch')
    arg_parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    arg_parser.add_argument('--wd', default=1e-5, type=float, help='Weight decay')
    arg_parser.add_argument('--save_model_interval', default=5, type=int, help='save model interval')
    
    args = arg_parser.parse_args()

    # Make directory.
    os.makedirs(args.model_path, exist_ok=True)

    # Validate paths.
    assert os.path.exists(args.data_csv)
    assert os.path.exists(args.model_path)

    return args


def main():
    # Parse arguments.
	args = parse_args()
	# Set device.
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set seed.
    seed = 42

    # Load image data
    image_df = pd.read_csv(args.data_csv, engine='python', header=None)
	image_df = image_df.reindex(np.random.permutation(image_df.index))
	num_valid = int(len(image_df) * 0.2)
	train_df = image_df[num_valid:]
	valid_df = image_df[:num_valid]

    print("train size: {}".format(len(train_df)))
    print("valid size: {}".format(len(valid_df)))
	
    # run training
    run_training(seed, train_df, valid_df, args)


if __name__ == "__main__":
	main()
    print("finished successfully.")
	os._exit(0)