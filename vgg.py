

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.autograd import Variable



epochs = 120
batch_size = 32
learning_rate = 0.01
num_workers = 4
encode_length = 365


torch.backends.cudnn.benchmark = True


model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, encode_length)

train_transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = ImageFolder('../../data/Places/train',transform=train_transform)
valset = ImageFolder('../../data/Places/val',transform=train_transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for epoch in epochs:
    if epoch in [30, 60, 90]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    model.train()
    for i, (img,label) in enumerate(trainloader):
        img = Variable(img.cuda())
        label = Variable(label.cuda())