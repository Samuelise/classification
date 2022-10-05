
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
learning_rate = 0.1
num_workers = 4
encode_length = 365


torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def adjust_learning_rate(optim, epoch):
    if epoch in [30,60,90]:
        for param_group in optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def main(checkpoint=''):
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
    valloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=num_workers)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()
    val_acc = evaluate(model, valloader, criterion)
    
    torch.save({'state_dict': model.state_dict}, "pths/vgg16_places365.pth")


def train(model, trainloader, optimizer, criterion, epoch):
    total = 0.
    correct = 0.
    loss_sum = 0.
    if epoch in [30, 60, 90]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    model.train()
    for i, (img,label) in enumerate(trainloader):
        img = Variable(img.cuda())
        label = Variable(label.cuda())

        output = model(img)
        loss = criterion(output, label)
        _, pred = torch.max(output.data, 1)
        correct += pred.eq(label.data).cpu().sum()
        total += label.size(0)
        loss_sum+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("train epoch %d/%d, prec1: %.3f, loss: %.3f" % epoch, epochs, correct/total, loss_sum/(i + 1))



def evaluate(model, valloader, criterion):
    total = 0.
    correct = 0.
    loss_sum = 0.
    model.eval()
    for i, (img, label) in enumerate(valloader):
        img = Variable(img.cuda())
        label = Variable(label.cuda())
        output = model(img)
        loss = criterion(output, label)
        _, pred = torch.max(output.data, 1)
        correct += pred.eq(label.data).cpu().sum()
        total += label.size(0)
        loss_sum+=loss
    print("valid prec1: %.3f loss: %.3f" % correct/total, loss_sum/(i + 1))

    return correct/total


if __name__ == "__main__":
    main()
    