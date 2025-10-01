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
            transforms.Resize((224,224)),                                   # original size 512x512,
            transforms.Grayscale(num_output_channels = 1),                   # converts all to grayscale (1 x 224 x 224), input for vit needs 3 channels
            transforms.ConvertImageDtype(torch.float32)                    # convert to torch tensor
            # transforms.ToTensor()
        ])

        self.ch3_transform = transforms.Compose([                          # second transform layer for images
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))                 # this converts 1ch to 3ch stacked (3C, H, W)
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

#*****************************************************************************************


# defining the model

model_id = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_id)
config = ViTConfig.from_pretrained(model_id)
VTmodel = ViTModel.from_pretrained(model_id)

# define model class

class VIT(nn.Module):

    def __init__(self, model_id, img_size, patch_size, freeze):
        super().__init__()

        # loading the pre-trained model
        self.config = ViTConfig.from_pretrained(model_id)
        self.vit = ViTModel.from_pretrained(model_id)      

        # define params
        self.img_size = 224                                 # default ViT input size, (original size: 512x512)
        self.patch_size = 16                                # default 16 patches
        self.num_patches = img_size // patch_size           # number of patches in image (14 patches)
        self.grid_size = (img_size // patch_size) ** 2      # grid size of each patch (14x14 = 196)

        # backbone vit encoder param freeze
        if freeze: 
            for par in self.vit.parameters():
                par.requires_grad = False                   # ViT params/weights frozen, only trains decoder
            print('encoder backbone frozen')
        else:
            print('encoder backbone training enabled')


        # define decoder 
        self.decoder = nn.Sequential(
            nn.LayerNorm(768),                                # input dim [B, grid_size = 196, hidden = 768]
            nn.Linear(768, 1024),                             # fc layer to [B, 196, 1024]
            nn.GELU(),                                        # gelu activation
            nn.Dropout(0.1),
            nn.Linear(1024, 512),                             # [B, 196, 512]
            nn.GELU(),                                        # gelu activation
            nn.Dropout(0.1),
            nn.Linear(512, 64),                               # [B, 196, 64]
        )
        

        # define spatial upsampling
        self.upsample = nn.Sequential(
            # nn.convTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            # dimensions [B, C, H, W]
            nn.ConvTranspose2d(64, 128, kernel_size = 4, stride = 2, padding = 1),         # [B, 64->128, 14->28, 14->28]
            nn.BatchNorm2d(128),                                                           # batch norm applied to 128 channels
            nn.ReLU(),  
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),         # [B, 128->64, 28->56, 28->56]
            nn.BatchNorm2d(64),                                                           # batch norm applied to 64 channels
            nn.ReLU(),  
            nn.ConvTranspose2d(64, 32, kernel_size = 4, stride = 2, padding = 1),         # [B, 64->32, 56->112, 56->112]
            nn.BatchNorm2d(32),                                                           # batch norm applied to 32 channels
            nn.ReLU(),  
            nn.ConvTranspose2d(32, 16, kernel_size = 4, stride = 2, padding = 1),         # [B, 32->16, 112->224, 112->224]
            nn.BatchNorm2d(16),                                                           # batch norm applied to 16 channels
            nn.ReLU(),  
            
            nn.Conv2d(16, 1, kernel_size = 1),                                            # [B, 1, 224, 224]
            )


        # define loss functions
        self.loss_bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.loss_dice = ml.dice.DiceLoss(sigmoid = True, squared_pred = True, reduction = 'mean', smooth_dr = 1e-5, smooth_nr = 1e-5)

        self.combined_loss = lambda y_pred, y_true: 0.7 * self.loss_bce(y_pred, y_true) + 0.3 * self.loss_dice(y_pred, y_true).mean()
        
        # combined weighted bce + dice loss
        self.loss_function = self.combined_loss

        # self.loss_function = nn.BCEWithLogitsLoss(reduction = 'mean')

    def forward(self, pixel_values, labels = None):
        encoder_outputs = self.vit(pixel_values, return_dict = True)        # hidden size 768, vit out dim [B, 196 + 1 cls token, 768]
        patch_embeddings = encoder_outputs.last_hidden_state[:, 1:, :]      # [batch, grid_size, hidden] indexed from 1 b/c CLS token prepended to grid_size at pos 0
        patch_features = self.decoder(patch_embeddings)                     # passes embeddings into decoder w/ shape [B, 196, 768] -> [B, 196, 64]
        
        batch_size = patch_features.shape[0]                                # returns batch size

        spatial_logits = patch_features.transpose(1, 2).reshape(
            batch_size, 64, self.num_patches, self.num_patches)              # transpose features [B, 196, 64] -> [B, 64, 196] and reshape to [B, C = 64, 14, 14]


        ups_logits = self.upsample(spatial_logits)                          # upsample to original size [B, 1, 224, 244]


        if labels is not None:                                              # if ground truth mask is provided (train/val)   
            labels = (labels > 0.5).float()
            loss = self.loss_function(ups_logits, labels)
            return {'loss': loss, "logits": ups_logits}                     # return loss and logits for training
        
        else: 
            return ups_logits                                               # return only logits


    def predict(self, pixel_values, threshold):
        with torch.no_grad():
            logits = self.forward(pixel_values)                             # computes logits in forward pass
            probas = torch.sigmoid(logits)                                  # computes probabilities from logits
            bin_msk = (probas > threshold).float()                          # creates binary mask w/ threshold value
        return bin_msk                                                      # bin_msk y pred
    
    def unfreeze(self):
        for par in self.vit.parameters():
            par.requires_grad = True                                         # unfreeze all ViT encoder backbone for full fine tuning
        print('vit encoder unfrozen')

    # def unfreeze_attn(self):
    #     n_layers = 2                                                        # number of layers to unfreeze
    #     t_layers = len(self.vit.encoder.layer)                              # total layers
    #     s_layer = max(0, t_layers - n_layers)

    #     for i in range(s_layer, t_layers):
    #         for par in self.vit.encoder.layer.attention.parameters():
    #             par.requires_grad = True
    #     print(f"last {n_layers} attention laters unfrozen")


#*****************************************************************************************


# define eval metrics

def eval_metrics(evalpred):
    logits = evalpred.predictions                           # returns model pred on val data
    y_true = evalpred.label_ids                             # returns gt mask on val data

    probas = 1 / (1 + np.exp(-logits))                      # sigmoid function
    y_pred = (probas > 0.5).astype(np.float32)              # convert to binary

    y_pred_fl = y_pred.ravel().astype(int)                  # flatten for sklearn
    y_true_fl = y_true.ravel().astype(int)

    y_pred_tensor = torch.from_numpy(y_pred.astype(np.float32))
    y_true_tensor = torch.from_numpy(y_true.astype(np.float32))
    
    # compute metrics 
    acc = accuracy_score(y_pred_fl, y_true_fl)
    f1 = f1_score(y_pred_fl, y_true_fl, zero_division = 0)
    prec = precision_score(y_pred_fl, y_true_fl, zero_division = 0)
    rec = recall_score(y_pred_fl, y_true_fl, zero_division = 0)
    js = jaccard_score(y_pred_fl, y_true_fl, zero_division = 0)

    # dice and iou scores
    dice_score = DiceScore(num_classes = 1, include_background = False)
    dice = dice_score(y_pred_tensor, y_true_tensor)
    
    meaniou = MeanIoU(num_classes = 1, include_background = False)
    miou = meaniou(y_pred_tensor, y_true_tensor)

    res_dict = {'acc': acc, 'dice': dice, 'mIoU': miou, 'f1': f1, 'rec': rec, 'prec': prec, 'jacc': js }

    return res_dict



# instatiate model
base_model = VIT(model_id, img_size = 224, patch_size = 16, freeze = True)

# send to gpu
torch.backends.mps.is_built()                       # check that mps build is compliant w/ pytorch

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(device)

base_model.to(device)

# define phased training

targs1 = TrainingArguments(
    output_dir = 'training/vit16/phase_1',                # separate training output
    num_train_epochs = 50,                               # training decoder, use more epochs
    per_device_train_batch_size = 25,
    per_device_eval_batch_size = 25,

    learning_rate = 1e-4 ,                              # try 5e-4 for bigger steps                      
    weight_decay = 0.01 ,

    # warmup_steps = 5,
    lr_scheduler_type='cosine',
    
    logging_strategy = 'epoch',
    save_strategy = 'epoch',
    eval_strategy = 'epoch',
    load_best_model_at_end = True,                      # trainer saves model
    metric_for_best_model = 'eval_loss',                # ['eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']
    greater_is_better = False,                          # false for BCE + dice loss, 
)


# instantiate trainer
trainer1 = Trainer(
    model = base_model,
    args = targs1,
    train_dataset = ds_train,
    eval_dataset = ds_val,
    compute_metrics = eval_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)] 
)

trainer1.train()


trainer1.state
trainer1.state.log_history
trainer1.state.best_model_checkpoint                    # returns best model chkpt
trainer1.state.best_metric                              # returns best metric


# phase 2

# reinstantiate
base_model_p2 = VIT(model_id, img_size = 224, patch_size = 16, freeze = True)

# load params
checkpoint = trainer1.state.best_model_checkpoint
# checkpoint = 'training/vit16/phase_1/checkpoint-1760'

# load safetensors file
w_path = checkpoint + '/model.safetensors'              

# load weights into model
state = load_file(w_path, device = 'cpu')          
base_model_p2.load_state_dict(state)

# unfreeze decoder
base_model_p2.unfreeze()

print(device)
base_model_p2.to(device)


targs2 = TrainingArguments(
    output_dir = 'training/vit16/phase_2',
    num_train_epochs = 5,
    learning_rate = 1e-4 ,
    weight_decay = 0.01 ,
    per_device_train_batch_size = 25,
    per_device_eval_batch_size = 25,

    # warmup_ratio=
    lr_scheduler_type = 'cosine',

    save_strategy = 'epoch',
    eval_strategy = 'epoch',
    logging_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_loss',
    greater_is_better = False,
)

trainer2 = Trainer(
    model = base_model_p2 ,
    args = targs2,
    train_dataset = ds_train, 
    eval_dataset = ds_val,
    compute_metrics = eval_metrics
)

trainer2.train()

trainer2.state
trainer2.state.log_history
trainer2.state.best_model_checkpoint                    # returns best model chkpt
trainer2.state.best_metric                              # returns best metric



#*****************************************************************************************

# define metrics function

# y_pred and y_true have size [N, 1, 224, 224]
def get_metrics(y_pred, y_true):

    # for sklearn
    y_pred_np = y_pred.numpy().ravel().astype(int)                  
    y_true_np = y_true.numpy().ravel().astype(int)

    y_pred_tensor = y_pred
    y_true_tensor = y_true
    
    # compute metrics 
    acc = accuracy_score(y_pred_np, y_true_np)
    f1 = f1_score(y_pred_np, y_true_np, zero_division = 0)
    prec = precision_score(y_pred_np, y_true_np, zero_division = 0)
    rec = recall_score(y_pred_np, y_true_np, zero_division = 0)
    js = jaccard_score(y_pred_np, y_true_np, zero_division = 0)

    # dice and iou scores
    dice_score = DiceScore(num_classes = 1, include_background = False)
    dice = dice_score(y_pred_tensor, y_true_tensor)
    
    meaniou = MeanIoU(num_classes = 1, include_background = False)
    miou = meaniou(y_pred_tensor, y_true_tensor)

    res = {'acc': acc, 'dice': dice, 'mIoU': miou, 'f1': f1, 'rec': rec, 'prec': prec, 'jacc': js }

    return res


# model results

# base model p1 -- only decoder trained
base_model.to(device = 'cpu')

y_pred = base_model.predict(test_x, threshold = 0.5)
y = test_y

get_metrics(y_pred, y)

color_sequences = ['viridis', 'plasma', 'inferno', 'magma', 'cividis'] 
for i in range(10):
    pred_mask = y_pred[i].squeeze().numpy()
    plt.imshow(pred_mask, cmap = 'viridis')
    plt.show()



# base model p2

base_model_p2.to(device = 'cpu')
y_pred = base_model_p2.predict(test_x, threshold = 0.5)
y = test_y

get_metrics(y_pred, y)

color_sequences = ['viridis', 'plasma', 'inferno', 'magma', 'cividis'] 
for i in range(10):
    pred_mask = y_pred[i].squeeze().numpy()
    plt.imshow(pred_mask, cmap = 'cividis', 
               vmax = 1.0, vmin = 0.0, interpolation = 'nearest')
    plt.show()






''' REF. DOCS
hf transformers.EvalPrediction (https://huggingface.co/docs/transformers/v4.56.2/en/internal/trainer_utils#transformers.EvalPrediction)
torchmetrics dicescore (https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html)
torchmetrics mean iou (https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html)
'''
