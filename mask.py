'''
In this code snippet, we will import the JSON data from the SYNTAX dataset 
and generate segmentation masks

[Reference]: codebase from original paper for mask generation
https://github.com/cmctec/ARCADE/blob/main/useful%20scripts/create_masks.ipynb

this code has been modified from the original to suite the purpose of our model
'''


#// packages & dependencies

import cv2
import json
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


#------------------------------------------------------------------------
#***************************************************************************************

#// loading data

print(os.getcwd())

# PATH = '../syntax/val/annotations/val.json'         # path to coco json annotation file
PATH = 'syntax/val/annotations/val.json'
N_IMG = 200                                           # number of images

#// reading annotation
with open(PATH, encoding='utf-8') as file:
    gt = json.load(file)


type(gt)
gt.keys()
gt['images']
gt['annotations'][0]

len(gt['annotations'])  # len 1168 because of categories

gt['annotations'][0]['image_id']

# idx_position = {img['id']: i for i, img in enumerate(gt['images'])}

image_annot_gt = defaultdict(list)                  # creates empty default dict

for ann in gt['annotations']:
    image_annot_gt[ann['image_id']].append(ann)

img_pos = sorted(image_annot_gt.keys())
img_id_index = {img_id: i for i, img_id in enumerate(img_pos)}


# define ground truth mask object
gt_mask = np.zeros((N_IMG, 512, 512), np.int32)           # 26 is number of categories 

for idx, img in image_annot_gt.items():
    for ann in img:
        points = np.array([ann['segmentation'][0][0::2], # all points of x
                           ann['segmentation'][0][1::2]],  # all points of y
                           np.int32).T 
        points = points.reshape((-1, 1, 2))

        tmp = np.zeros((512,512), np.int32)
        cv2.fillPoly(tmp, [points], (1))

        # gt_mask[idx - 1, ann['category_id']] += tmp
        # gt_mask[idx - 1, ann['category_id'], gt_mask[idx - 1, ann['category_id']] > 0 ] = 1

        # gt_mask[img_id_index[idx]][tmp == 1] = ann['category_id']      # this is an aggregation by category id 
        gt_mask[img_id_index[idx]][tmp == 1] = 1      # this is an aggregation by category id 

plt.imshow(gt_mask[1])              # gt_mask[i] is a mask of image i for all categories
plt.show()


# plot the image
from PIL import Image
IMG_PATH = 'syntax/val/images/'
img = Image.open(IMG_PATH + '2.png')

img.mode       # L -- luminance/grayscale
img.size

plt.imshow(img)
plt.show()

gt_mask.shape


mask1 = gt_mask[1]
mask1.shape



gt['images'] # stores filenames relative to index id
filenames = [img['file_name'] for img in gt['images']]

len(filenames)

# convert masks and save as each as 

MASK_PATHS = ['syntax/val/masks', 'syntax/train/masks', 'syntax/test/masks']


def save_masks(dataset):
    for i in range(gt_mask.shape[0]):

        MASK_PATHS = ['syntax/val/masks', 'syntax/train/masks', 'syntax/test/masks']    

        if dataset == 'val':
            OUT_PATH = MASK_PATHS[0]
        
        elif dataset == 'train':
            OUT_PATH = MASK_PATHS[1]
        
        elif dataset == 'test':
            OUT_PATH = MASK_PATHS[2]

        mask = gt_mask[i-1]

        mask = (mask.astype(np.uint8)) * 255        # convert to 8bit and grayscale

        filename = f'{filenames[i-1]}'                     # defines indexed filenames

        output = os.path.join(OUT_PATH, filename)

        Image.fromarray(mask).save(output)          # save to png in output dir 

        print(f'mask {i+1} successfully saved to: {OUT_PATH}')  



save_masks('val')



# refactor mask code into function    

ANN_PATHS = ['syntax/val/annotations/val.json', 'syntax/train/annotations/train.json', 'syntax/test/annotations/test.json']

def generate_masks(dataset):
    if dataset == 'val':
        PATH = ANN_PATHS[0]
    
    elif dataset == 'train':
        PATH = MASK_PATHS[1]
     
    elif dataset == 'test':
        PATH = MASK_PATHS[2]


    with open(PATH, encoding='utf-8') as file:
        gt = json.load(file)
    
    
    # gt filenames here

    image_annot_gt = defaultdict(list)                  # creates empty default dict

    for ann in gt['annotations']:
        image_annot_gt[ann['image_id']].append(ann)     # image_annot_gt is a dictionary of lists w/ 200 keys




    



