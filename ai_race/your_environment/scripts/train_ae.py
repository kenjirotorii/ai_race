import os
import sys
import cv2
import random
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from autoencoder import Autoencoder, VAE, VAELoss


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


class AEDataSet(Dataset):
    def __init__(self, path_list, transdorm=None, crop=True):
        """DataSet for Autoencoder

        Args:
            path_list (list): image file names
            transdorm ([type], optional): [description]. Defaults to None.
        """
        self.path_list = path_list
        self.transform = transdorm
        self.crop = crop

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        input_paths = [BASE_PATH + s + "/images/" + self.path_list[idx] for s in SEASON]
        inputs = [get_img(path, self.crop) for path in input_paths]
        target_path = BASE_PATH + "normal/images/" + self.path_list[idx]
        target = get_img(target_path, self.crop)
        
        if self.transform:
            inputs = [self.transform(img.copy()) for img in inputs]
            target = self.transform(target.copy())

        return inputs, target


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device, variational):

    model.train()
    final_loss = 0

    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        for img in inputs:
            img = img.to(device)
            if variational:
                y_pred, mu, lnvar = model(img)
                loss = loss_fn(y_pred, targets, mu, lnvar)
            else:
                y_pred = model(img)
                loss = loss_fn(y_pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss += loss.item() / len(inputs)
        
        scheduler.step()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device, variational):

    model.eval()
    final_loss = 0
    
    for _, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.to(device)

        for img in inputs:
            img = img.to(device)
            if variational:
                y_pred, mu, lnvar = model(img)
                loss = loss_fn(y_pred, targets, mu, lnvar)
            else:
                y_pred = model(img)
                loss = loss_fn(y_pred, targets)

            final_loss += loss.item() / len(inputs)

    final_loss /= len(dataloader)

    return final_loss


def inference_and_save(model, file_name=None, device="cpu", save_path=None, crop=True, variational=False, epoch=0):

    model.eval()

    if file_name is None:
        return

    if save_path is None:
        save_path = BASE_PATH + "tmp"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    epoch_path = save_path + "/epoch_" + str(epoch)
    if not os.path.exists(epoch_path):
        os.mkdir(epoch_path)
    
    styles = SEASON + ["normal"]
    for s in styles:
        img_path = BASE_PATH + s + "/images/" + file_name
        img = get_img(img_path, crop)
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
        path = epoch_path + "/" + file_name.split('.')[0] + "_pred_" + s + ".jpg"
        cv2.imwrite(path, im_bgr)

    img_path = BASE_PATH + "normal/images/" + file_name
    path = epoch_path + "/" + file_name
    original = get_img(img_path, crop)
    im_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im_bgr)


def run_training(seed, train_path, valid_path, variational=False):
    
    seed_everything(seed)
    
    train_dataset = AEDataSet(train_path, transdorm=transforms.ToTensor(), crop=IMG_CROP)
    valid_dataset = AEDataSet(valid_path, transdorm=transforms.ToTensor(), crop=IMG_CROP)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if variational:
        model = VAE(h=120, w=320, outputs=NUM_Z)
        loss_fn = VAELoss()
        model_name = "vae"
    else:
        model = Autoencoder(h=120, w=320, outputs=NUM_Z)
        loss_fn = nn.BCELoss()
        model_name = "ae"
    
    model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
        
    best_loss = np.inf
    best_model_path = model_name + "_best.pth"

    for epoch in range(EPOCHS):
        
        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE, variational)
        valid_loss = valid_fn(model, loss_fn, validloader, DEVICE, variational)
        print("EPOCH: {}, train_loss: {}, valid_loss: {}".format(epoch, train_loss, valid_loss))
        
        if valid_loss < best_loss:
            
            best_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
        
        if epoch % 5 == 4:
            model_path = model_name + "_ckpt_{}.pth".format(epoch + 1)
            torch.save(model.state_dict(), model_path)

        # inference test data
        if epoch % INF_INTERVAL == 0:
            inference_path = random.sample(valid_path, INF_NUM)
            for path in inference_path:
                inference_and_save(model, path, DEVICE, SAVE_PATH, IMG_CROP, variational, epoch)


def main():
    # image file paths
    img_file_path = glob.glob(BASE_PATH + "normal/images/*.jpg")
    img_file_path = [path.split("/")[-1] for path in img_file_path]

    # split train and valid
    random.shuffle(img_file_path)               # shuffle path list
    train_num = int(0.8 * len(img_file_path))   # number of train data
    train_path = img_file_path[:train_num]      # paths of train data
    valid_path = img_file_path[train_num:]      # paths of valid data
    
    print("train size: {}".format(len(train_path)))
    print("valid size: {}".format(len(valid_path)))

    # run training
    run_training(SEED, train_path, valid_path, VARIATIONAL)


if __name__ == "__main__":
    
    # config
    BASE_PATH = "/home/jetson/"
    SAVE_PATH = BASE_PATH + "test_vae"
    SEASON = ["spring", "summer", "autumn", "winter"]
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # parameter
    NUM_Z = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e0
    EPOCHS = 25
    SEED = 42
    INF_INTERVAL = 1
    INF_NUM = 10
    IMG_CROP = True
    VARIATIONAL = True
    
    # main function
    main()
