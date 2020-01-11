from sklearn.metrics import f1_score, roc_auc_score
from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os
from datahandle_celeb import get_dataloader_sep_folder
import matplotlib.pyplot as plt
from PIL import Image

batch_size = 6


def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=False, progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    model.train()
    return model


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
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
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                if inputs.shape == torch.Size([batch_size, 3, 256, 256]):
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'Train'):
                        outputs = model(inputs)
                        loss = criterion(outputs['out'], masks)
                        y_pred = outputs['out'].data.cpu().numpy().ravel()
                        y_true = masks.data.cpu().numpy().ravel()

                        for name, metric in metrics.items():
                            if name == 'f1_score':
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true > 0, y_pred > 0.1))
                            else:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true.astype('uint8'), y_pred))
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
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model

model = createDeepLabv3()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
data_dir= "data_celeb"
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
dataloaders = get_dataloader_sep_folder(data_dir, imageFolder='Images', maskFolder='Masks', batch_size=batch_size)
train_model(model, criterion, dataloaders, optimizer, metrics, "metrics", num_epochs=6)
torch.save(model.state_dict(), os.path.join("metrics", 'weights.pt'))