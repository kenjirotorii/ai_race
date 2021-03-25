import os
import random

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path, crop=True):
    img = cv2.imread(path)
    if crop:
        h = img.shape[0]
        img = img[int(h/2):, :, :]
        img = cv2.resize(img, (160, 80))
    return img


def calc_score(output_list, target_list, running_loss, data_loader):
    # Calculate accuracy.
    acc = round(f1_score(output_list, target_list, average='micro'), 6)
    loss = round(running_loss / len(data_loader.dataset), 6)
    return acc, loss
