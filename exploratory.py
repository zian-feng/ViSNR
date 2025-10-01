# packages & dependencies
import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from transformers import ViTModel, AutoConfig, ViTImageProcessor, ViTConfig

from transformers import Trainer, TrainingArguments, AutoModelForSemanticSegmentation, AutoTokenizer
from transformers import EarlyStoppingCallback

from safetensors.torch import load_file

import monai.losses as ml               # monai loss functions library
import monai.metrics as mm              # monai metrics library

from torchmetrics.segmentation import DiceScore, MeanIoU

from monai.metrics import meandice
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, jaccard_score

#*****************************************************************************************

# loading data
ABS_PATH = '/Users/daofeng/Desktop/______/INM363/CODE/syntax'               # ABS path of syntax df

# define custom data class
class Syntax():
    def __init__(self, root_path, dataset):
        
        self.root_path =  root_path

        if dataset == 'train':
            
            self.images = sorted([root_path + '/train/images/' + i for i in os.listdir(root_path + '/train/images/') if not i.startswith('.')],
                                 key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            
            self.masks  = sorted([root_path + '/train/masks/' + i for i in os.listdir(root_path + '/train/masks/') if not i.startswith('.')],
                                 key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))            
            
            self.labels = sorted(os.listdir(root_path + '/train/images/'), key = lambda x: int(os.path.splitext(x)[0]))
            # self.annots = root_path + '/train/annotations/' + 'train.json'

        elif dataset == 'val':

            self.images = sorted([root_path + '/val/images/' + i for i in os.listdir(root_path + '/val/images/') if not i.startswith('.')],
                                 key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            
            self.masks  = sorted([root_path + '/val/masks/' + i for i in os.listdir(root_path + '/val/masks/') if not i.startswith('.')],
                                 key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            
            self.labels = sorted(os.listdir(root_path + '/val/images/'), key = lambda x: int(os.path.splitext(x)[0]))

        elif dataset == 'test':

            self.images = sorted([root_path + '/test/images/' + i for i in os.listdir(root_path + '/test/images/') if not i.startswith('.')],
                                 key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            
            self.masks  = sorted([root_path + '/test/masks/' + i for i in os.listdir(root_path + '/test/masks/') if not i.startswith('.')],
                                 key = lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            self.labels = sorted(os.listdir(root_path + '/test/images/'), key = lambda x: int(os.path.splitext(x)[0]))

        else:
            raise ValueError("dataset parameter needs to be 'train', 'val', or 'test' ")
            # pass

        
        self.transform = transforms.Compose([
            # transforms.Resize((224,224)),                                  # original size 512x512,
            transforms.Grayscale(num_output_channels = 1),                 # converts all to grayscale (1 x 224 x 224), input for vit needs 3 channels
            transforms.ConvertImageDtype(torch.float32)                    # convert to torch tensor
        ])

        self.ch3_transform = transforms.Compose([                          # second transform layer for images
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))                 # this converts 1ch to 3ch stacked (3C, H, W)
        ])

        self.msk_transform = transforms.Compose([
            # transforms.Resize((224,224),
            #                   interpolation = InterpolationMode.NEAREST),                                  # original size 512x512
            transforms.ConvertImageDtype(torch.float32),                    # convert to torch tensor
            transforms.Lambda(lambda pixel: (pixel > 0).float())            # binarize masks
        ])


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        img = read_image(self.images[idx])                  # loads images
        img = self.transform(img)                           # applies transforms
        img = self.ch3_transform(img)                       # converts to 3 channels



        msk = read_image(self.masks[idx])                   # loads masks
        msk = self.transform(msk)                           # applies transforms

        lbl = self.labels[idx]
        
        return img, msk, lbl

# initialize datasets
data_train = Syntax(root_path = ABS_PATH, dataset = 'train')
data_val = Syntax(root_path = ABS_PATH, dataset = 'val')
data_test = Syntax(root_path = ABS_PATH, dataset = 'test')


# define data loaders
BATCHN = [10, 20, 25, 50, 100]                       # different batch sizes batch sizes

train_loader = DataLoader(dataset = data_train, shuffle = False, batch_size = BATCHN[2])
test_loader = DataLoader(dataset = data_test, shuffle = False, batch_size = BATCHN[3])
val_loader = DataLoader(dataset = data_val, shuffle = False, batch_size = BATCHN[4])

train_imgs, train_msks, train_lbls = next(iter(train_loader))
val_imgs, val_msks, val_lbls = next(iter(val_loader))

#****************************************************************************************************************

# load full train/val/test data from dataloaders
data_train
data_val
data_test

def load_all(loader):
    imgs, msks, lbls = [], [], []

    for i, m, l in loader:
        imgs.append(i)
        msks.append(m)
        lbls.append(l)
    
    imgs = torch.cat(imgs, dim = 0)             # concatenate batches
    msks = torch.cat(msks, dim = 0)

    return imgs, msks, lbls


train_imgs, train_msks, train_lbls = load_all(train_loader)
val_imgs, val_msks, val_lbls = load_all(val_loader)
test_imgs, test_msks, test_lbls = load_all(test_loader)

# use processor and collator to prepare the images
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model_input = processor(images = train_imgs, return_tensors = 'pt')

# processing each dataset for trainer
train_x = processor(images = train_imgs, return_tensors = 'pt')
train_x = train_x['pixel_values']
train_y = train_msks

val_x = processor(images = val_imgs, return_tensors = 'pt')
val_x = val_x['pixel_values']
val_y = val_msks

test_x = processor(images = test_imgs, return_tensors = 'pt')
test_x = test_x['pixel_values']
test_y = test_msks

# collate into one dataset dict of X, y
class SYN(Dataset):
    def __init__(self, pixel_values: torch.Tensor, masks: torch.Tensor):
        self.x = pixel_values
        self.y = masks
    
    def __len__(self):
        return self.y.size(0)
    
    def __getitem__(self, idx):
        dat = {
            'pixel_values': self.x[idx],
            'labels': self.y[idx]
        }
        
        return dat

ds_train = SYN(train_x, train_y)
ds_val = SYN(val_x, val_y)
ds_test = SYN(test_x, test_y)

#****************************************************************************************


# basic count data

# calculate mask background vs foreground ratio



# plot images & masks

# export random sample of train images in luminance


# examine some edge cases

# analyse edge cases


# pixel counts by bg and fg ration

train_y.shape
val_y.shape
test_y.shape




fg_pixels = []

for msk in train_y:
    fg_count = (msk.squeeze(0) == 1).sum()
    fg_count = fg_count.item()

    fg_pixels.append(fg_count)


len(fg_pixels)
fg_pixels 
tot_pixels = 512 * 512

fg_percent = [f / tot_pixels for f in fg_pixels]

max(fg_percent)
min(fg_percent)

np.mean(fg_percent)*100             # 0.019627232142857144
100 - np.mean(fg_percent)*100

(3.68 / 96.32) 
1 / (96.32 / 3.68)

'''
this means that in our training data we have very imbalanced pixel ratio
for the foreground and background

we will make proportional changes in weighting of our loss function to alleviate this
'''



# visualizing filters

# exporting selected samples with luminance

preview_imgs_idx = [1, 11, 19, 33, 93]
preview_imgs_idx = [idx - 1 for idx in preview_imgs_idx]

preview_imgs_idx

preview_imgs = train_imgs[preview_imgs_idx]

for idx, img in enumerate(preview_imgs):
    img = img[0].numpy()

    dpi = 100
    plt.figure(figsize = (512/dpi, 512/dpi), dpi = dpi)
    plt.imshow(img, cmap = 'viridis')
    plt.axis('off')
    plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
    # plt.savefig(f'preview img {idx}.png')

