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

dataloaders = get_dataloader_sep_folder("data_test", imageFolder='Images', maskFolder='Masks', batch_size=1)

model = createDeepLabv3(3)
model.load_state_dict(torch.load("metrics/weights.pt"))
model.eval()

for i in iter(dataloaders["Train"]):
    input_tensor = i["image"][0]
    # print(input_tensor.shape)
    # input_tensor_img = i["image"][0]

    input_batch = input_tensor.unsqueeze(0)
    print(input_batch.shape)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
        output = output['out'][0]
        print(output.shape)

    output_predictions = output.argmax(0)

    print(output_predictions.shape)
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    # r2 = Image.fromarray(input_tensor_img.byte().cpu().numpy())
    # criterion = torch.nn.MSELoss(reduction='mean')
    # print(criterion(i["image"].to("cuda:0"),input_batch.to("cuda:0")))
    import matplotlib.pyplot as plt
    print(output_predictions.byte().cpu().numpy().shape)
    plt.imshow(r)
    plt.text(35,-20,i['slika'][0])
    plt.show()


