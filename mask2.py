'''
In this code snippet, we will import the JSON data from the SYNTAX dataset 
and generate segmentation masks

[Reference]: codebase from original paper for mask generation
https://github.com/cmctec/ARCADE/blob/main/useful%20scripts/create_masks.ipynb

this code has been modified from the original to suite the purpose of our model
'''

# packages & dependencies

import cv2
import json
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from PIL import Image

#***************************************************************************************

# loading data
print(os.getcwd())

# path to coco json annotation file
ANN_PATHS = ['syntax/val/annotations/val.json', 'syntax/train/annotations/train.json', 'syntax/test/annotations/test.json']
PATH = ANN_PATHS[1]

# PATH = 'syntax/val/annotations/val.json'
# N_IMG = 200                                           # number of images



# reading annotation coco json file
with open(PATH, encoding='utf-8') as file:
    gt = json.load(file)


image_annot_gt = defaultdict(list)                  # creates empty default dict

for ann in gt['annotations']:
    image_annot_gt[ann['image_id']].append(ann)     # image_annot_gt is a dictionary of lists w/ 200 keys


# define ground truth mask object
N_IMG = len(image_annot_gt)
gt_mask = np.zeros((N_IMG, 512, 512), np.int32)           # drop 26 (num. of categories) as a dim 

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

        # gt_mask[idx - 1][tmp == 1] = ann['category_id']      # this is an aggregation by category id 
        gt_mask[idx - 1][tmp == 1] = 1                         # this is an aggregation by category id per image





# plot the mask
plt.imshow(gt_mask[1])              # gt_mask[i] is a mask of image i for all categories (200x512x512)
plt.show()

# plot the image
IMG_PATH = 'syntax/test/images/'
img = Image.open(IMG_PATH + '1.png')

img.mode       # L -- luminance/grayscale
img.size

plt.imshow(img)
plt.show()


# extract filenames

''' Note
gt['images'][i][file_names] has the filename of each mask and it is not in order, 
thus when we export the masks we will name in order of the aggregated filenames.
'''

gt['images']
gt['images'][1].keys()                             # stores filenames relative to index id
filenames = [img['file_name'] for img in gt['images']]

print(filenames)

# convert masks and save as each as png

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

        mask = gt_mask[i]

        mask = (mask.astype(np.uint8)) * 255                # convert to 8bit and grayscale

        filename = f'{filenames[i]}'                        # defines indexed filenames

        # filename = f'{i+1}.png'                               # use for train set

        output = os.path.join(OUT_PATH, filename)

        Image.fromarray(mask).save(output)                  # save to png in output dir 

        print(f'mask {i+1} successfully saved to: {OUT_PATH}')  


# Functions for saving masks
# save_masks('val')
# save_masks('test')
# save_masks('train')

'''
Due to the way that the raw image data was saved when published, each subset of the data (train, val, test) 
had been indexed differently. Thus, the code needed modification for each instance.
'''


''' Code References:
[1] [ARCADE Scripts](https://github.com/cmctec/ARCADE/blob/main/useful%20scripts/create_masks.ipynb)
[2]
[3]
'''