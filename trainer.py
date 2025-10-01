# packages & dependencies
import os

from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import ViTModel, AutoConfig, ViTImageProcessor, ViTConfig

from transformers import Trainer, TrainingArguments, AutoModelForSemanticSegmentation, AutoTokenizer


import monai.losses as ml               # monai loss functions library
import monai.metrics as mm              # monai metrics library

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

#****************************************************************************************************************

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

        # backbone param freeze
        if freeze: 
            for par in self.vit.parameters():
                par.requires_grad = False                   # ViT params/weights frozen, only trains decoder
            print('encoder backbone frozen')
        else:
            print('encoder backbone training enabled')

        # define decoder
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),          # layernormalization to hidden size
            nn.Linear(self.config.hidden_size, 512),        # hidden size -> 512
            nn.GELU(),                                      # gelu layer
            nn.Dropout(0.1),                                # add dropout
            nn.Linear(512, 256),                            # linear layer 512 -> 256
            nn.GELU(),                                      # gelu layer
            nn.Dropout(0.1),                                # add dropout
            nn.Linear(256, 1)                               # linear layer 256 -> 1
        )
        
        # define upsample interpolation         
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1, out_channels = 32, kernel_size = 3, 
                               stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 3, 
                               stride = 2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3, 
                               stride = 2, padding = 1, output_padding = 1)
        )

    def forward(self, pixel_values):
        encoder_outputs = self.vit(pixel_values, return_dict = True)        # hidden size 768
        patch_embeddings = encoder_outputs.last_hidden_state[:, 1:, :]      # [batch, grid_size, hidden] CLS token prepended to grid_size
        patch_logits = self.decoder(patch_embeddings)                       # passes embeddings into decoder w/ shape [batch, grid_size, hidden] -> [batch, grid_size, 1]
        
        b_size = patch_logits.shape[0]

        spatial_logits = patch_logits.transpose(1, 2).reshape(
            b_size, 1, self.num_patches, self.num_patches)                  # converts to [B, C = 1, 14, 14]
        
        upsampled_logits = self.upsample(spatial_logits)                    # upsample to original size [B, 1, 224, 244]
        
        return upsampled_logits

    def predict(self, pixel_values, threshold):
        with torch.no_grad():
            logits = self.forward(pixel_values)                             # computes logits in forward pass
            probas = torch.sigmoid(logits)                                  # computes probabilities from logits
            bin_msk = (probas > threshold).float()                          # creates binary mask w/ threshold value
        return bin_msk                                                      # bin_msk y pred
    
    def unfreeze(self):
        for par in self.vit.parameters():
            par.require_grad = True                                         # unfreeze ViT encoder backbone for fine tuning
        print('vit encoder unfrozen')

    def unfreeze_attn(self):
        n_layers = 2                                                        # number of layers to unfreeze
        t_layers = len(self.vit.encoder.layer)                              # total layers
        s_layer = max(0, t_layers - n_layers)

        for i in range(s_layer, t_layers):
            for par in self.vit.encoder.layer.attention.parameters():
                par.requires_grad = True
        print(f"last {n_layers} attention laters unfrozen")



# loss functions
dice_loss = ml.DiceLoss(                        # takes 'upsampled_logits' var as input
    sigmoid = True,                             # use sigmoid function for logits
    squared_pred = True,                        # std. dice coeff
    reduction = 'mean',                         # avg loss for batch
    smooth_dr = 1e-5,                           # denominator smoothing to avoid NaNs 
    smooth_nr = 1e-5                            # numerator smoothing to avoid zero values
    )

gendice_loss = ml.GeneralizedDiceLoss(
    include_background = True,
    to_onehot_y = False, 
    sigmoid = False,
    softmax = False,
)


jaccard_loss = ml.DiceLoss(
    sigmoid = True,
    squared_pred = False,
    jaccard = True,             # computes jaccard index
    reduction = 'mean',
    smooth_dr = 1e-5,           # adds constant to denominator to avoid NANs
    smooth_nr = 1e-5            # adds constant to numerator to avoid zeros
)


# metrics

dice_metric = mm.DiceMetric(
    include_background = False,
    reduction = 'mean',
    get_not_nans = False
)

iou_metric = mm.MeanIoU(
    include_background = False,
    reduction = 'mean'
)

# f-score, sensitivity, specificity, recall
# import from sklearn

metrics = {
    'dice': dice_metric,
    'iou' : iou_metric
}

mm.confusion_matrix                 # function to retrieve confusion matrix
mm.ConfusionMatrixMetric()          # stateful: can aggregate on forward pass

cmat = mm.confusion_matrix()

mm.generalized_dice()

# define compute metrics
# def metrics(eval_pred):
#     predictions = eval_pred                 # pass in bin_mask from forward pass in ViT
#     ...


# define hyper parameter searching
# make sure to use random search for efficiency


# instantiate model

base_model = VIT(model_id, img_size = 224, patch_size = 16, freeze = True)


# define hyper parameter space
hp = []

Trainer.hyperparameter_search()



# define training

# define training arguments
train_args = TrainingArguments(
    output_dir = 'phase_1',                             # separate training output
    num_train_epochs = 10,                              # training decoder, use more epochs
    learning_rate = 1e-5 ,                              # 
    weight_decay = 0.05 ,
    load_best_model_at_end = True,
    metric_for_best_model = 'dice',
    greater_is_better = 'True',

)


# instantiate trainer
trainer1 = Trainer(
    model = ... ,
    args = train_args,
    train_dataset = train_imgs, # include binary masks
    compute_metrics = metrics
)

# trainer.train()

# model.unfreeze()                              # unfreezes encoder backbone
# model.unfreeze_attn()                         # unfreezes attention layers

# phase 2

train_args_p2 = TrainingArguments(
    output_dir = 'phase_2',
    num_train_epochs = ...,
    learning_rate = ... ,
    weight_decay = ... ,
    load_best_model_at_end = True,
    metric_for_best_model = 'dice',
    greater_is_better = 'True',

)

trainer2 = Trainer(
    model = ... ,
    args = train_args_p2,
    train_dataset = train_imgs, # include binary masks
    compute_metrics = metrics
)

# trainer.train()


# training
# trainer.train()
# results = trainer.evaluate()



torch.backends.mps.is_available()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device
model = base_model

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
criterion = ml.DiceLoss()


type(ml.DiceLoss())
type(torch.nn.CrossEntropyLoss())


# trainer.hyperparameter_search()



print(enumerate(train_loader))


'''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,     # note: processor handles images
)
'''
