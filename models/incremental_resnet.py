import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_cifar import resnet32


class Incremental_ResNet(nn.Module):
    def __init__(self, starting_classes=10, cosine=True):
        super(Incremental_ResNet, self).__init__()
        self.cosine = cosine

        self.backbone = resnet32(num_classes=starting_classes)
        self.feat_size = self.backbone.out_dim
        if self.cosine:
            self.fc1 = nn.Linear(self.feat_size, starting_classes, bias=False)
        else:
            self.fc1 = nn.Linear(self.feat_size, starting_classes, bias=True)
        self.eta = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.backbone(x)  # get features

        if self.cosine:
            x_norm = F.normalize(x, p=2, dim=1)
            with torch.no_grad():
                self.fc1.weight.div_(torch.norm(self.fc2.weight, dim=1, keepdim=True))
            y = self.eta * self.fc1(x_norm)
        else:
            y = self.fc1(x)

        return x, y

    def expand_classes(self, new_classes):

        old_classes = self.fc2.weight.data.shape[0]
        # print(old_classes)
        old_weight = self.fc2.weight.data
        # self.n_classes += new_classes
        self.fc1 = nn.Linear(self.feat_size, old_classes + new_classes, bias=False)
        self.fc1.weight.data[:old_classes] = old_weight

    def classify(self, x):

        if self.cosine:
            x_norm = F.normalize(x, p=2, dim=1)  # FIXME *T radius can be T not 1
            with torch.no_grad():
                self.fc1.weight.div_(torch.norm(self.fc2.weight, dim=1, keepdim=True))
            y = self.eta * self.fc1(x_norm)

        return x, y


def ResNet32Cifar(starting_classes=10, cosine=True):
    model = Incremental_ResNet(starting_classes, cosine)
    return model