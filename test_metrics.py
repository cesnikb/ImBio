import numpy as np
import torch
from PIL import Image

from datahandle import get_dataloader_sep_folder
import matplotlib.pyplot as plt
data_dir= "data_test"
dataloaders = get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=4)
criterion = torch.nn.MSELoss(reduction='mean')

class unNormalize(object):
    '''Normalize image'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': image.type(torch.FloatTensor)*255,
                'mask': mask.type(torch.FloatTensor)*255}

for o,a in enumerate(iter(dataloaders["Train"])):
    i = a.unNormalize()
    input = i["image"][0]
    input_m = i["mask"][0]
    input_c = input.byte().cpu().numpy()
    input_b = np.transpose(input_c, (1, 2, 0))
    input_cmask = input_m.byte().cpu().numpy()
    input_bmask = np.transpose(input_cmask, (1, 2, 0))
    r = Image.fromarray(input_b,"RGB")
    rmask = Image.fromarray(input_bmask,"RGB")
    plt.imshow(r)
    plt.text(35, -20, i['slika'][0])
    plt.show()
    plt.imshow(rmask)
    plt.text(35, -20, i['slika'][0])
    plt.show()
    # mask = i["mask"][0]
    # print(mask)
    # r = Image.fromarray(input.byte().cpu().numpy())

# inpit = a["image"].to("cuda:0")
# inpit2 = b["image"].to("cuda:0")
# print(criterion(inpit,inpit2))
