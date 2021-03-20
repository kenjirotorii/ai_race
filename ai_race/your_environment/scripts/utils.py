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
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    if crop:
        h = im_rgb.shape[0]
        im_rgb = im_rgb[int(h/2):, :, :]
        
    return im_rgb


def calc_score(output_list, target_list, running_loss, data_loader):
    # Calculate accuracy.
    acc = round(f1_score(output_list, target_list, average='micro'), 6)
    loss = round(running_loss / len(data_loader.dataset), 6)
    return acc, loss