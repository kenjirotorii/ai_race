'''
Train autoencoder.
'''
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

from autoencoder import Autoencoder, VAE, VAELoss
from utils import get_img, seed_everything
from train_funcs import train_ae, valid_ae


HOME_PATH = os.environ['HOME']
CWD_PATH = os.path.dirname(os.path.abspath(__file__))

def inference_and_save(model, args, file_name=None, device="cpu", variational=True, epoch=0):

    model.eval()

    if file_name is None:
        return

    epoch_path = args.result_path + "epoch_" + str(epoch)
    if not os.path.exists(epoch_path):
        os.mkdir(epoch_path)
    
    for s in ["spring", "summer", "autumn", "winter", "normal"]:
        img_path = "{}/{}/images/{}".format(HOME_PATH, s, file_name)
        img = get_img(img_path, crop=True)
        img = transforms.ToTensor()(img.copy())
        img = img.unsqueeze(0)
        img = img.to(device)
        
        if variational:
            outputs, _, _ = model(img)
            outputs = outputs.detach().cpu().squeeze().numpy()
        else:
            outputs = model(img).detach().cpu().squeeze().numpy()
        
        outputs = 255*outputs.transpose((1, 2, 0))

        im_bgr = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
        path = "{}/{}_pred_{}.jpg".format(epoch_path, file_name.split('.')[0], s)
        cv2.imwrite(path, im_bgr)

    img_path = "{}/normal/images/{}".format(HOME_PATH, file_name)
    path = epoch_path + "/" + file_name
    original = get_img(img_path, crop=True)
    im_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im_bgr)


def run_training(seed, train_path, valid_path, variational, device, args):

    seed_everything(seed)
    
    train_dataset = AEDataSet(train_path, transdorm=transforms.ToTensor(), crop=True)
    valid_dataset = AEDataSet(valid_path, transdorm=transforms.ToTensor(), crop=True)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    if variational:
        model = VAE(h=120, w=320, outputs=args.num_z)
        loss_fn = VAELoss()
        model_name = "vae"
    else:
        model = Autoencoder(h=120, w=320, outputs=args.num_z)
        loss_fn = nn.BCELoss()
        model_name = "ae"
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=args.n_epochs, steps_per_epoch=len(trainloader))
        
    best_loss = np.inf
    best_model_path = args.model_path + args.model_name + "_best.pth"

    for epoch in range(args.n_epoch):
        
        train_loss = train_ae(model, optimizer, scheduler, loss_fn, trainloader, device, variational)
        valid_loss = valid_ae(model, loss_fn, validloader, device, variational)
        print("EPOCH: {}, train_loss: {}, valid_loss: {}".format(epoch, train_loss, valid_loss))
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
        
        if epoch % args.save_model_interval == 4:
            model_path = "{}_ckpt_{}.pth".format(args.model_path + args.model_name, epoch + 1)
            torch.save(model.state_dict(), model_path)

        # inference test data
        if epoch % args.inf_model_interval == 0:
            inference_path = random.sample(valid_path, args.num_inf)
            for path in inference_path:
                inference_and_save(model, args, path, device, variational, epoch)


def parse_args():

    arg_parser = argparse.ArgumentParser(description="Autoencoder")

    arg_parser.add_argument("--model_name", type=str, default='control_model')
    arg_parser.add_argument("--model_path", type=str, default=CWD_PATH+'/models/')
    arg_parser.add_argument("--result_path", type=str, default=HOME_PATH+'/test_vae/')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--num_z', default=128, type=int, help='The number of latent variables')
    arg_parser.add_argument('--n_epoch', default=25, type=int, help='The number of epoch')
    arg_parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate')
    arg_parser.add_argument('--wd', default=1e-5, type=float, help='Weight decay')
    arg_parser.add_argument('--save_model_interval', default=5, type=int, help='save model interval')
    arg_parser.add_argument('--inf_model_interval', default=1, type=int, help='inference interval')
    arg_parser.add_argument('--num_inf', default=10, type=int, help='The number of inferenced image sets')
    
    args = arg_parser.parse_args()

    # Make directory.
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    # Validate paths.
    assert os.path.exists(args.model_path)
    assert os.path.exists(args.result_path)

    return args


def main():
    # Parse arguments.
    args = parse_args()
    # Set device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set seed.
    seed = 42

    # image file paths
    img_file_path = glob.glob(HOME_PATH + "/normal/images/*.jpg")
    img_file_path = [path.split("/")[-1] for path in img_file_path]

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
