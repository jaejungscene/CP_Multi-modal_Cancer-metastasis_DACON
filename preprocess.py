import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

def make_tiles(img, tile_size=256, num_tiles= 4):
    '''
    img: np.ndarray with dtype np.uint8 and shape (width, height, channel)
    '''
    w, h, ch = img.shape
    pad0, pad1 = (tile_size - w%tile_size) % tile_size, (tile_size - h%tile_size) % tile_size
    padding = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
    img = np.pad(img, padding, mode='constant', constant_values=255)
    img = img.reshape(img.shape[0]//tile_size, tile_size, img.shape[1]//tile_size, tile_size, ch)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, ch)
    if len(img) < num_tiles: # pad images so that the output shape be the same
        padding = [[0, num_tiles-len(img)], [0, 0], [0, 0], [0, 0]]
        img = np.pad(img, padding, mode='constant', constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_tiles] # pick up Top N dark tiles
    img = img[idxs]
    return img

# def get_transforms(args.action):
# train_transforms = A.Compose([
#                             A.HorizontalFlip(),
#                             A.VerticalFlip(),
#                             A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
#                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
#                             ToTensorV2()
#                             ])

# test_transforms = A.Compose([
#                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
#                             ToTensorV2()
#                             ])