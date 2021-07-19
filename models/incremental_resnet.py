import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.resnet_cifar import resnet32
from models.resnet_imagenet import resnet18
from models.CosineClassifier import CosineClassifier
from models.my_resnet import resnet_rebuffi
class Incremental_ResNet(nn.Module):
    def __init__(self,backbone="resnet32", starting_classes=10, cosine=True):
        super(Incremental_ResNet, self).__init__()
        self.cosine = cosine
        self.backbone = resnet_rebuffi()
        # if backbone == "resnet32":
        #     self.backbone = resnet32(num_classes=starting_classes)
        # elif backbone =="resnet18":
        #     self.backbone = resnet18(pretrained=False, num_classes=starting_classes)
        self.feat_size = self.backbone.out_dim
        #self.fc1 = nn.Linear(self.feat_size, starting_classes, bias=not(self.cosine))
        if self.cosine:
            self.fc1 = CosineClassifier(self.feat_size, starting_classes)
        else:
            self.fc1 = nn.Linear(self.feat_size, starting_classes)

    def forward(self, x):
        x = self.backbone(x)  # get features

        y = self.fc1(x)

        return x, y

    def expand_classes(self, new_classes):

        old_classes = self.fc1.weight.data.shape[0]
        old_weight = self.fc1.weight.data
        if self.cosine:
            self.fc1 = CosineClassifier(self.feat_size, old_classes + new_classes)
        else:
            self.fc1 = nn.Linear(self.feat_size, old_classes + new_classes)
        self.fc1.weight.data[:old_classes] = old_weight


    def classify(self, x):
        y = self.fc1(x)
        return x, y
        
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
