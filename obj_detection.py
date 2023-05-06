
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import csv
import os
import shutil
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
import copy
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
import random
from torchvision.models import vgg16

dirrs = ["images_01.tar\\images_01\\images","images_02.tar\\images_02\images","images_03.tar\\images_03\\images","images_04.tar\\images_04\\images","images_05.tar\\images_05\\images",
"images_06.tar\\images_06\\images","images_07.tar\\images_07\\images","images_08.tar\\images_08\\images","images_09.tar\\images_09\\images","images_10.tar\\images_10\\images",
"images_11.tar\\images_11\\images","images_12.tar\\images_12\\images"]

bbxes = []
with open('BBox_List_2017.csv',newline='') as csvfile:
    rreader = csv.reader(csvfile, delimiter=',')
    next(rreader)
    for row in rreader:
        bbxes.append(row)

nms = []
for bx in bbxes:
    nms.append(bx[0])
img_dir = "imgs"

def is_in(fn, names):
    for name in names:
        if fn == name:
            return True
    return False

for dirr in dirrs:
    for filename in os.listdir(dirr):
        if is_in(filename, nms):
            f = os.path.join(dirr, filename)
            shutil.copy(f, "imgs\\")

def find_coords(filename):
    for i in range(len(bbxes)):
        # bounding box defined by upper left and lower right coordinates
        if bbxes[i][0] == filename:
            coords = [float(bbxes[i][2]), float(bbxes[i][3]), \
            float(bbxes[i][2]) + float(bbxes[i][4]), \
            float(bbxes[i][3]) + float(bbxes[i][5])]
            # normalize
            for p in range(len(coords)):
                coords[p] /= 1024
            return np.array(coords)

labs = []
for filename in os.listdir(img_dir):
    labs.append([find_coords(filename), filename])


class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][1])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = self.img_labels[idx][0]
        label = torch.from_numpy(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image.float(), label.float()


class MyNet(nn.Module):
    def __init__(self):
        torch.manual_seed(2023)
        random.seed(2023)
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)

        self.fcn1 = nn.Linear(7 * 7 * 512, 2048)
        self.fcn2 = nn.Linear(2048, 128)
        self.fcn3 = nn.Linear(128, 32)
        self.fcn4 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.pool(x)
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        x = self.pool(F.relu(self.conv10(F.relu(self.conv9(F.relu(self.conv8(x)))))))
        x = self.pool(F.relu(self.conv13(F.relu(self.conv12(F.relu(self.conv11(x)))))))
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(F.relu(self.fcn1(x)))
        x = self.dropout(F.relu(self.fcn2(x)))
        x = self.dropout(F.relu(self.fcn3(x)))
        x = self.fcn4(x)

        return x

def iou(lab, pred):
    avg_iou = 0
    for i in range(len(lab)):
        cur_lab = lab[i]
        cur_pred = pred[i]
        x1 = cur_lab[0]
        y1 = cur_lab[1]
        x2 = cur_pred[0]
        y2 = cur_pred[1]
        wlab = cur_lab[2] - cur_lab[0]
        hlab =  cur_lab[3] - cur_lab[1]
        wpred = cur_pred[2] - cur_pred[0]
        hpred = cur_pred[3] - cur_pred[1]

        if x1 < x2:
            if x1 + wlab < x2 + wpred: # if box doesn't surpass edge of other box in x coords
                width_intersection = x1 + wlab - x2
            else:
                width_intersection = wpred
        else:
            if x2 + wpred < x1 + wlab:
                width_intersection = x2 + wpred - x1
            else:
                width_intersection = wlab
        if y1 < y2:
            if y1 + hlab < y2 + hpred:
                height_intersection = y1 + hlab - y2
            else:
                height_intersection = hpred
        else:
            if y2 + hpred < y1 + hlab:
                height_intersection = y2 + hpred - y1
            else:
                height_intersection = hlab

        if width_intersection <= 0 or height_intersection <= 0:
            IoU = 0
        else:
            intersection = height_intersection * width_intersection
            union = wlab*hlab + wpred*hpred - intersection
            IoU = intersection / union
        avg_iou += IoU
    avg_iou /= len(lab)
    return avg_iou

def visualize_results(ins, labs, preds):
    dirr = "eval_ims"
    for i in range(len(ins)):
        ext = str(i) + ".png"
        img = ins[i]
        lab = labs[i].cpu().numpy()
        pred = preds[i].cpu().numpy()
        img = img.cpu().numpy()
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        img = np.ascontiguousarray(img).copy()
        img *= 256
        lab *= 224
        pred *= 224
        img = cv2.rectangle(img, (int(lab[0]), int(lab[1])), (int(lab[2]),int(lab[3])), (0, 0, 0), 5)
        img = cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]),int(pred[3])), (255, 255, 255), 5)
        cv2.imwrite(os.path.join(dirr, ext), img)

batch_size = 8
lr = 0.01

random.seed(2023)
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = MyDataset(labs, img_dir, transform)

training_size = int(0.8 * len(dataset))
test_size = len(dataset) - training_size
training_data, test_data = random_split(dataset, [training_size, test_size])

trainloader = DataLoader(training_data, batch_size=batch_size)
testloader = DataLoader(test_data, batch_size = test_size)

max_IoU = 0
best_weights = None
net = MyNet()
device = "cuda:0"
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
# training
num_not_improved = 0
for epoch in range(100):
    if torch.cuda.is_available():
        net.to(device)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 4:
        print(epoch)
    # testing
    with torch.no_grad():
        net.to("cpu")
        for data in testloader:
            inputs, labels = data
            pred = net(inputs)
            IoU = iou(labels, pred) # evaluation metric: IoU
            if IoU > max_IoU:
                print(epoch, IoU)
                max_IoU = IoU
                best_weights = copy.deepcopy(net.state_dict())
                num_not_improved = 0
                visualize_results(inputs, labels, pred)
            else:
                num_not_improved += 1

    if num_not_improved >= 10:
        break

net.load_state_dict(best_weights)
pat = '.\\unet.pth'
torch.save(net.state_dict(), pat)

