import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pc
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms as TF
from torch.utils.data import DataLoader

from ImageDataSet import ImageDataSet
from model import Model
from predict import *
import torch

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

csv_path = "..\\clip_image\\train_data\\Dataset.csv"
data_frame = pd.read_csv(csv_path)

def generate_labels(data, save=True):
    i = 1
    encoding = {}
    decodings = {}
    for each_data in data:
        encoding[each_data] = i
        decodings[i] = each_data
        i += 1
    if save:
        with open('.\\labels\\encodings.json', 'w') as fptr:
            json.dump(encoding, fptr, indent=4)
        with open('.\\labels\\decodings.json', 'w') as fptr:
            json.dump(decodings, fptr, indent=4)
    return encoding, decodings


label_mapper, label_inverse_mapper = generate_labels(data_frame['label'].unique().tolist())
data_frame['label'] = data_frame['label'].map(label_mapper)

def collate_fn(batch):
    return tuple(zip(*batch))

transform = TF.Compose([
    TF.Resize((224, 224)),
    TF.ToTensor()
])
dataset_coc = ImageDataSet(data_frame, transform)

def plot_image_with_bbox(image, bbox):
    plt.axes()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    for i in range(len(bbox['boxes'])):
        box = bbox['boxes'][i]
        x1, y1 = box[0], box[1]
        x2, y2 = box[2], box[3]
        rectangle = pc.Rectangle((x1,y1), x2-x1, y2-y1, fc='none',ec='red')
        plt.gca().add_patch(rectangle)
    plt.show()

dataloader = DataLoader(dataset=dataset_coc,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=collate_fn)

#Model
model = Model(out_classes = len(label_mapper) + 1)
model = model.to(device)
save_path = ".\\save\\basic_model.pt"
try:
    model.load_state_dict(torch.load(save_path))
except:
    pass

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
best_class_loss = best_regress_loss = np.inf;

epoch = 50
model.train()
best_class_loss = best_regress_loss = np.inf;
print('Starting Model Training')
for epoch in range(epoch):
    epoch_classif_loss = epoch_regress_loss = cnt = 0
    for batch_x, batch_y in dataloader:
        batch_x = list(image.to(device) for image in batch_x)
        batch_y = [{k: v.to(device) for k, v in t.items()} for t in batch_y]
        optimizer.zero_grad()
        loss_dict = model(batch_x, batch_y)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        epoch_classif_loss += loss_dict['loss_classifier'].item()
        epoch_regress_loss += loss_dict['loss_box_reg'].item()
        cnt += 1
    epoch_classif_loss /= cnt
    epoch_regress_loss /= cnt
    print("Training loss for epoch {} is {} for classification and {} for regression "
        .format(epoch + 1, epoch_classif_loss, epoch_regress_loss)
    )
    if (epoch_classif_loss <= best_class_loss) and (epoch_regress_loss <= best_regress_loss):
        print("*"*20)
        print(f"Classification Loss has reduced from {best_class_loss} to {epoch_classif_loss}")
        print(f"Regression Loss has reduced from {best_regress_loss} to {epoch_regress_loss} saving model.state")
        print("*"*20)
        torch.save(model.state_dict(), save_path)
        best_class_loss = epoch_classif_loss
        best_regress_loss = epoch_regress_loss

# predict(model, r'D:\GitHub\coc_bot\src\base_downloader\layouts\th5\5ea938b9213d4304ad84301c.jpg', transform, label_inverse_mapper)
