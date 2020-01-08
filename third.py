import urllib

from gluoncv import data
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn

from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os
from datahandle import get_dataloader_single_folder, get_dataloader_sep_folder
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

batch_size = 6


def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

def display_output(outputs):
    output = outputs['out'][0]
    input_batch = output.to('cuda')
    model.to('cuda')
    output_predictions = output.argmax(0)
    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")
    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    # r.putpalette(colors)
    plt.imshow(r)
    plt.show()


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
                 [f'Train_{m}' for m in metrics.keys()] + \
                 [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # print(inputs.shape)
                if inputs.shape == torch.Size([batch_size, 3, 218, 178]):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # track history if only in Train
                    with torch.set_grad_enabled(phase == 'Train'):

                        outputs = model(inputs)
                        # print(outputs['out'].shape)

                        # display_output(outputs)



                        loss = criterion(outputs['out'], masks)
                        y_pred = outputs['out'].data.cpu().numpy().ravel()
                        y_true = masks.data.cpu().numpy().ravel()

                        for name, metric in metrics.items():
                            if name == 'f1_score':
                                # Use a classification threshold of 0.1
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true > 0, y_pred > 0.1))
                            else:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true.astype('uint8'), y_pred))

                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(
                phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = createDeepLabv3(3)

# print_picture(model)
# model.train()

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
data_dir= "data"
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
dataloaders = get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=batch_size)
train_model(model, criterion, dataloaders, optimizer, metrics, "metrics", num_epochs=5)
torch.save(model.state_dict(), os.path.join("metrics", 'weights.pt'))

# print(1)
# model.eval()
# for i in iter(dataloaders["Train"]):
#     a = model(i["image"])
#     plt.imshow(a)
#     plt.show()
#
# import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
#
# from PIL import Image
# from torchvision import transforms
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)


# model.eval()
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')
# with torch.no_grad():
#     output = model(input_batch)['out'][0]
# output_predictions = output.argmax(0)
#
# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")
#
# # plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
# r.putpalette(colors)
#
# import matplotlib.pyplot as plt
# plt.imshow(r)
# plt.show()

# train_model(model, criterion, dataloaders, optimizer, metrics, "metrics", num_epochs=3)
# torch.save(model, os.path.join("metrics", 'weights.pt'))