import cv2
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

import models
from datahandle_celeb import Resize, ToTensor, Normalize
from models.segmentation.deeplabv3 import DeepLabHead
import matplotlib.pyplot as plt


def createDeepLabv3(outputchannels=1,pretrained=True):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=pretrained, progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    return model


def print_img(filename):
    image = cv2.imread(filename, 1).transpose(2, 0, 1)
    transform = transforms.Compose([Resize((256,256),(256,256)),ToTensor(), Normalize()])
    a =dict()
    a["image"] = image
    a["mask"] = image
    transformed_pict = transform(a)
    input_batch = transformed_pict["image"].unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
        output = output['out'][0]
    t = Variable(output)  # threshold
    out = (0.5 > t).float() * 1
    out = out[0]
    output_numpy = out.byte().cpu().numpy()

    ra = Image.fromarray(output_numpy)
    plt.imshow(ra)
    plt.show()


model = createDeepLabv3(1,False)
model.load_state_dict(torch.load("metrics/weights_notpretrained.pt"))
model.eval()

print_img('bojan.jpg')
