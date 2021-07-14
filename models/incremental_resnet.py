import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.resnet_cifar import resnet32
from models.resnet_imagenet import resnet18

class Incremental_ResNet(nn.Module):
    def __init__(self,backbone="resnet32", starting_classes=10, cosine=True):
        super(Incremental_ResNet, self).__init__()
        self.cosine = cosine
        if backbone == "resnet32":
            self.backbone = resnet32(num_classes=starting_classes)
        elif backbone =="resnet18":
            self.backbone = resnet18(pretrained=False, num_classes=starting_classes)
        self.feat_size = self.backbone.out_dim
        self.fc1 = nn.Linear(self.feat_size, starting_classes, bias=not(self.cosine))
        # if self.cosine:
        #     self.fc1 = nn.Linear(self.feat_size, starting_classes, bias=False)
        # else:
        #     self.fc1 = nn.Linear(self.feat_size, starting_classes, bias=True)
        self.eta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def forward(self, x):
        x = self.backbone(x)  # get features

        if self.cosine:
            x_norm = F.normalize(x, p=2, dim=1)
            with torch.no_grad():
                self.fc1.weight.div_(torch.norm(self.fc1.weight, dim=1, keepdim=True))
            y = self.eta * self.fc1(x_norm)
        else:
            y = self.fc1(x)

        return x, y

    def expand_classes(self, new_classes):

        old_classes = self.fc1.weight.data.shape[0]
        # print(old_classes)
        old_weight = self.fc1.weight.data
        # self.n_classes += new_classes
        self.fc1 = nn.Linear(self.feat_size, old_classes + new_classes, bias=not(self.cosine))
        if self.eta is not None:
             self.eta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
        self.fc1.weight.data[:old_classes] = old_weight


    def classify(self, x):

        if self.cosine:
            x_norm = F.normalize(x, p=2, dim=1)  # FIXME *T radius can be T not 1
            with torch.no_grad():
                self.fc1.weight.div_(torch.norm(self.fc1.weight, dim=1, keepdim=True))
            y = self.eta * self.fc1(x_norm)

        return x, y

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc1.weight.data.size(1))
        self.fc1.weight.data.uniform_(-stdv, stdv)
        if self.eta is not None:
            self.eta.data.fill_(1) 

def ResNet32Incremental(starting_classes=10, cosine=True):
    model = Incremental_ResNet(backbone="resnet32",starting_classes=starting_classes, cosine=cosine)
    return model

def ResNet18Incremental(starting_classes=10, cosine=True):
    model = Incremental_ResNet(backbone="resnet18",starting_classes=starting_classes, cosine=cosine)
    return model


__factory = {
    'imagenet': "resnet18",
    'cifar100': "resnet32"
}

def create(dataset, classes, cosine):
    if dataset not in __factory.keys():
        raise KeyError(f"Unknown Model: {dataset}")
    return Incremental_ResNet(__factory[dataset], starting_classes=classes, cosine=cosine)

    