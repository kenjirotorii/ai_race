import os
import random
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from networks.recognet import RecogNet
from common.utils import seed_everything
from common.train_funcs import train_fn, valid_fn
from common.make_datasets import RecogDataSet


HOME_PATH = os.environ['HOME']


def run_training(seed, train_path, valid_path, device, args):

    seed_everything(seed)
    
    train_dataset = RecogDataSet(train_path, transform=transforms.ToTensor())
    valid_dataset = RecogDataSet(valid_path, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = RecogNet(80, 160, 4)
    loss_fn = nn.CrossEntropyLoss()
    
    model.to(device)
    
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
        
        if (epoch + 1) % args.save_model_interval == 0:
            model_path = "{}_ckpt_{}.pth".format(args.model_path + args.model_name, epoch + 1)
            torch.save(model.state_dict(), model_path)


def parse_args():

    arg_parser = argparse.ArgumentParser(description="Season Recognition")

    arg_parser.add_argument("--model_name", type=str, default='season_recog')
    arg_parser.add_argument("--model_path", type=str, default=HOME_PATH+'/catkin_ws/src/ai_race/ai_race/your_environment/models/')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--n_epoch', default=25, type=int, help='The number of epoch')
    arg_parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    arg_parser.add_argument('--wd', default=0.0, type=float, help='Weight decay')
    arg_parser.add_argument('--save_model_interval', default=5, type=int, help='save model interval')
    
    args = arg_parser.parse_args()

    # Make directory.
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Validate paths.
    assert os.path.exists(args.model_path)

    return args


def main():
    # Parse arguments.
    args = parse_args()
    # Set device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set seed.
    seed = 42

    # image file paths
    img_file_path = []
    for season in ["spring", "summer", "autumn", "winter"]:
        img_file_path += glob.glob("{}/{}/images/*.jpg".format(HOME_PATH, season))

    # split train and valid
    random.shuffle(img_file_path)               # shuffle path list
    train_num = int(0.8 * len(img_file_path))   # number of train data
    train_path = img_file_path[:train_num]      # paths of train data
    valid_path = img_file_path[train_num:]      # paths of valid data
    
    print("train size: {}".format(len(train_path)))
    print("valid size: {}".format(len(valid_path)))

    # run training
    run_training(seed, train_path, valid_path, device, args)


if __name__ == "__main__":
    main()
    print("finished successfully.")
    os._exit(0)
