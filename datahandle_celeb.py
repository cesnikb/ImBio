from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
import cv2
import torch
from torchvision import transforms, utils
import imageio
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)

# def one_hot(img):
#     img = np.array(img)
#     (np.arange(np.max(img)) == img[...,None]-1).astype(int)
def one_hot(a):
    ncols = a.max()+1
    out = np.zeros(a.shape + (ncols,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out

class CelebDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder, transform=None,  imagecolormode='rgb', maskcolormode='grayscale'):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Masks
            -----Masks 1
            -----Masks N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed: Specify a seed for the Train and test split
            fraction: A float value from 0 to 1 which specifies the validation split fraction
            subset: 'Train' or 'Test' to select the appropriate set.
            imagecolormode: 'rgb' or 'grayscale'
            maskcolormode: 'rgb' or 'grayscale'
        """
        self.imageFolder=imageFolder
        self.color_dict = {'rgb': 1, 'grayscale': 0}
        assert(imagecolormode in ['rgb', 'grayscale'])
        assert(maskcolormode in ['rgb', 'grayscale'])
        self.mapping = {
            0: 0,
            255: 1
        }
        self.imagecolorflag = self.color_dict[imagecolormode]
        self.maskcolorflag = self.color_dict[maskcolormode]
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted(
            glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
        self.mask_names = sorted(
            glob.glob(os.path.join(self.root_dir, maskFolder, '*')))

    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask

    def __len__(self):
        return len(self.mask_names)



    def __getitem__(self, idx):
        msk_name = self.mask_names[idx]
        img_name = os.path.join(self.root_dir, self.imageFolder, str(int(msk_name.split("/")[3].split("_")[0]))+".jpg")
        image = cv2.imread(img_name, self.imagecolorflag).transpose(2, 0, 1)
        #novo
        mask = cv2.imread(msk_name,0)
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        sample['slika'] = img_name
        return sample


class Resize(object):
    """Resize image and/or masks."""

    def __init__(self, imageresize, maskresize):
        self.imageresize = imageresize
        self.maskresize = maskresize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, self.maskresize, cv2.INTER_AREA)
        image = cv2.resize(image, self.imageresize, cv2.INTER_AREA)
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2, 0, 1)

        return {'image': image,
                'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        image, mask = sample['image'], sample['mask']
        if len(mask.shape) == 2:
            mask = mask.reshape((1,)+mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class Normalize(object):
    '''Normalize image'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)/255
                }


def get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=4):
    """
        Create Train and Test dataloaders from two separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Images
        ---------Image1
        ---------ImageN
        ------Masks
        ---------Mask1
        ---------MaskN
        --Train
        ------Images
        ---------Image1
        ---------ImageN
        ------Masks
        ---------Mask1
        ---------MaskN
    """
    data_transforms = {
        'Train': transforms.Compose([Resize((256,256),(256,256)),ToTensor(), Normalize()]),
        'Test': transforms.Compose([Resize((256,256),(256,256)),ToTensor(), Normalize()]),
    }
    image_datasets = {
        x: CelebDataset(root_dir=os.path.join(data_dir, x), transform=data_transforms[x], maskFolder=maskFolder,
                      imageFolder=imageFolder)
        for x in ['Train', 'Test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=8)
                   for x in ['Train', 'Test']}
    return dataloaders
