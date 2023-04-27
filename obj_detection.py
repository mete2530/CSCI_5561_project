
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd



#!/usr/bin/env python3
# Download the 56 zip files in Images_png in batches
import urllib.request

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

for idx, link in enumerate(links):
    fn = 'images_%02d.tar.gz' % (idx+1)
    print('downloading'+fn+'...')
    urllib.request.urlretrieve(link, fn)  # download the zip file

print("Download complete. Please check the checksums")


class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class MyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=2)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=2)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=2)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=2)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=2)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=2)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=2)

        self.upconv1 = nn.ConvTranspose2d(512, 256, 3, padding=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 3, padding=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 3, padding=2)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 3, padding=2)

        self.conv11 = nn.Conv2d(512, 256, 3, padding=2)
        self.conv12 = nn.Conv2d(256, 256, 3, padding=2)
        self.conv13 = nn.Conv2d(256, 128, 3, padding=2)
        self.conv14 = nn.Conv2d(128, 128, 3, padding=2)
        self.conv15 = nn.Conv2d(128, 64, 3, padding=2)
        self.conv16 = nn.Conv2d(64, 64, 3, padding=2)
        self.conv17 = nn.Conv2d(64, 32, 3, padding=2)
        self.conv18 = nn.Conv2d(32, 32, 3, padding=2)

        self.conv19 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        feat1 = F.relu(self.conv2(F.relu(self.conv1(x))))
        feat2 = F.relu(self.conv4(F.relu(self.conv3(self.pool(feat1)))))
        feat3 = F.relu(self.conv6(F.relu(self.conv5(self.pool(feat2)))))
        feat4 = F.relu(self.conv8(F.relu(self.conv7(self.pool(feat3)))))
        x = F.relu(self.conv10(F.relu(self.conv9(self.pool(feat3)))))

        x = self.upconv1(x)
        x = torch.cat((x, feat4), dim=1)
        x = self.upconv2(F.relu(self.conv12(F.relu(self.conv11(x)))))
        x = torch.cat((x, feat3), dim=1)
        x = self.upconv3(F.relu(self.conv14(F.relu(self.conv13(x)))))
        x = torch.cat((x, feat2), dim=1)
        x = self.upconv4(F.relu(self.conv16(F.relu(self.conv15(x)))))
        x = torch.cat((x, feat1), dim=1)
        x = F.relu(self.conv18(F.relu(self.conv17(x))))
        x = nn.sigmoid(self.conv(19))

        return x

net = MyUNet()
criterion = nn.BCELoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=0.01)

# training
for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# testing
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
