import torch
from PIL import Image

import models
from datahandle import get_dataloader_sep_folder
from models.segmentation.deeplabv3 import DeepLabHead
import matplotlib.pyplot as plt

def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

dataloaders = get_dataloader_sep_folder("data_small", imageFolder='Images', maskFolder='Masks', batch_size=4)

model = createDeepLabv3(3)
model.load_state_dict(torch.load("metrics/weights.pt"))
model.eval()

for i in iter(dataloaders["Train"]):
    input_tensor = i["image"][0]
    input_tensor_img = i["image"][0]
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)


    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    # r2 = Image.fromarray(input_tensor_img.byte().cpu().numpy())

    import matplotlib.pyplot as plt

    plt.imshow(r)
    plt.show()
    print(1)


