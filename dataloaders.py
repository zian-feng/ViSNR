# packages & dependencies
import os
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader

from transformers import ViTModel

import matplotlib.pyplot as plt

torch.backends.mps.is_available()

from PIL import Image

#****************************************************************************************************************

# import training data
ABS_PATH = '/Users/daofeng/Desktop/______/INM363/CODE/syntax'               # ABS path of syntax df
IMG_PATH = 'syntax/train/images/'
ANN_PATH = 'syntax/train/annotations/test.json'

# defining custom class for dataloaders using pytorch dataloaders

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
            pass

        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),                                   # original size 512x512,
            transforms.Grayscale(num_output_channels = 1),                   # converts all to grayscale (1 x 224 x 224)
            transforms.ConvertImageDtype(torch.float32)                    # convert to torch tensor
            # transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        img = read_image(self.images[idx])                  # loads images
        img = self.transform(img)                           # applies transforms

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

# define all three loaders

train_loader = DataLoader(dataset = data_train, shuffle = False, batch_size = BATCHN[2])
test_loader = DataLoader(dataset = data_test, shuffle = False, batch_size = BATCHN[3])
val_loader = DataLoader(dataset = data_val, shuffle = False, batch_size = BATCHN[4])



# previewing train data
images, masks, labels = next(iter(train_loader))

fig, ax = plt.subplots(5, 5, figsize = (10, 10) )

for i in range(BATCHN[2]):
    # display image
    ax[i // 5, i % 5].imshow(images[i].permute(1,2,0))               # note permute function is (H,W,C)
    ax[i // 5, i % 5].set_title(f'image {labels[i]}')
    ax[i // 5, i % 5].axis('off')

    # # display masks
    # ax[1+i, i].imshow(masks[i].permute(1,2,0))
    # ax[1+i, i].set_title(f'image {labels[i]}')
    # ax[1+i, i].axis('off')

plt.tight_layout()
plt.show()


# load the data using custom data loaders

for img, msk, _ in train_loader():
    print(img, msk)


type(images)
type(masks)
type(labels)

print(labels)

print(masks)

images.shape
masks.shape



# preview selected samples




# pulling models from huggingface

train = IMG_PATH 



# importing data

ABS_PATH = 'syntax'


# check that images are in luminance or grayscale
img1 = Image.open(IMG_PATH + '1.png')
img1.mode       # L -- luminance/grayscale
img1.size

plt.imshow(img1)
plt.show()


img2 = Image.open(IMG_PATH + '2.png')
img2.mode       # RGB 

img2 = img2.convert('L')
img2.size

plt.imshow(img2, cmap = 'plasma')
# add title / label
plt.show()

color_sequences = ['viridis', 'plasma', 'inferno', 'magma', 'cividis'] 

plt.imshow(img2, cmap = color_sequences[2])
# add title / label
plt.show()

