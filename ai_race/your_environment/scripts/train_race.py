'''
Train vechile control with camera images
'''
import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from autoencoder import VAE, Autoencoder, ControlHead
from utils import seed_everything
from train_funcs import train_fn, valid_fn
from make_datasets import ControlDataSet


HOME_PATH = os.environ['HOME']
CWD_PATH = os.path.dirname(os.path.abspath(__file__))


def run_training(seed, train_df, valid_df, device, args):

    seed_everything(seed)
    
    train_dataset = ControlDataSet(train_df, transform=transforms.ToTensor(), crop=True)
    valid_dataset = ControlDataSet(valid_df, transform=transforms.ToTensor(), crop=True)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.variational:
        model = VAE(h=80, w=160, outputs=args.num_z, train_mode=False)
    else:
        model = Autoencoder(h=80, w=160, outputs=args.num_z)

    model.load_state_dict(torch.load(args.pretrained_model))

    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name == "encoder":
            param.requires_grad = False

    model.decoder = ControlHead(args.num_z, 3)
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=args.n_epoch, steps_per_epoch=len(trainloader))
        
    best_loss = np.inf
    best_model_path = args.model_path + args.model_name + "_best.pth"

    for epoch in range(args.n_epoch):
        
        train_acc, train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, device)
        valid_acc, valid_loss = valid_fn(model, loss_fn, validloader, device)
        print("EPOCH: {}, train_loss: {:.6f}, valid_loss: {:.6f}, train_acc:{}, valid_acc:{}".format(epoch, train_loss, valid_loss, train_acc, valid_acc))
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
        
        if epoch % args.save_model_interval == 4:
            model_path = "{}_ckpt_{}.pth".format(args.model_path + args.model_name, epoch + 1)
            torch.save(model.state_dict(), model_path)


def parse_args():

    arg_parser = argparse.ArgumentParser(description="Image Classification")
    
    arg_parser.add_argument("--variational", action='store_true')
    arg_parser.add_argument("--data_path", type=str, default=HOME_PATH+'/Images_from_rosbag/')
    arg_parser.add_argument("--model_name", type=str, default='control_model')
    arg_parser.add_argument("--model_path", type=str, default=CWD_PATH+'/models/')
    arg_parser.add_argument("--pretrained_model", type=str, default=CWD_PATH+'/models/vae_mse_ckpt_25.pth')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--num_z', default=256, type=int, help='The number of latent variables')
    arg_parser.add_argument('--n_epoch', default=25, type=int, help='The number of epoch')
    arg_parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    arg_parser.add_argument('--wd', default=0.0, type=float, help='Weight decay')
    arg_parser.add_argument('--save_model_interval', default=5, type=int, help='save model interval')
    
    args = arg_parser.parse_args()

    # Make directory.
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Validate paths.
    assert os.path.exists(args.data_path)
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
    csv_paths = glob.glob(args.data_path + "*/*.csv")
    image_df = pd.DataFrame()
    for path in csv_paths:
        _df = pd.read_csv(path, engine='python', header=None)
        image_df = pd.concat([image_df, _df])
    image_df.index = range(len(image_df))
    image_df = image_df.reindex(np.random.permutation(image_df.index))
    num_valid = int(len(image_df) * 0.2)
    train_df = image_df[num_valid:]
    valid_df = image_df[:num_valid]

    print("train size: {}".format(len(train_df)))
    print("valid size: {}".format(len(valid_df)))
	
    # run training
    run_training(seed, train_df, valid_df, device, args)


if __name__ == "__main__":
    main()
    print("finished successfully.")
    os._exit(0)
