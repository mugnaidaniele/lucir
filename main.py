import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from models import create
from continuum.datasets import CIFAR100
from continuum import ClassIncremental
from train import train
from validate import validate
from utils.ExemplarSet import ExemplarSet
from utils.utils import get_all_images_per_class
from torch.optim.lr_scheduler import MultiStepLR
from utils.feature_selection import perform_selection
from utils.utils import getTransform, get_dataset
from torchvision.transforms import transforms

import os
import numpy as np
import copy

scheduling = [80, 120]

parser = argparse.ArgumentParser(description='Learning unified classifier via rebalancing')
parser.add_argument('--dataset', type=str, default="cifar100", metavar='BATCH', help='batch size')
parser.add_argument('--start', type=int, default=50, metavar='INC', help='increment classes')
parser.add_argument('--increment', type=int, default=50, metavar='INC', help='increment classes')
parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH', help='batch size')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate for cnn')
parser.add_argument('--momentum', type=float, default=0.9 , metavar='LR', help='momentum')
parser.add_argument('--epochs', type=int, default=1, metavar='BATCH', help='batch size')
parser.add_argument('--lamda_base', type=float, default=5.0, metavar='LR', help='learning rate for cnn')
parser.add_argument('--margin', type=float, default=0.2, metavar='LR', help='learning rate for cnn')
parser.add_argument('--gamma', type=float, default=0.1, metavar='LR', help='learning rate for cnn')
parser.add_argument('--k-negatives', type=int, default=2, metavar='BATCH', help='batch size')
parser.add_argument('--rehearsal', type=int, default=20, metavar='BATCH', help='batch size')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='LR', help='learning rate for cnn')

parser.add_argument('--selection', type=str, default="herding", metavar='BATCH', help='batch size')
parser.add_argument("--exR", action="store_true", default=True, help="experience replay")
parser.add_argument("--cosine", action="store_true", default=True, help="experience replay")
parser.add_argument("--less_forg", action="store_true", default=True, help="experience replay")
parser.add_argument("--ranking", action="store_true", default=True, help="experience replay")

args = parser.parse_args()
model = create("cifar100", args.start, args.cosine)
model.cuda()
dataset_train, dataset_test = get_dataset(args.dataset)
train_transform, val_transform, test_transform = getTransform(args.dataset)

scenario_train = ClassIncremental(dataset_train, increment=args.increment, initial_increment=args.start, transformations=train_transform)
scenario_val = ClassIncremental(dataset_test, increment=args.increment, initial_increment=args.start, transformations=val_transform)
assert scenario_train.nb_tasks == scenario_val.nb_tasks

print(f"Number of classes: {scenario_train.nb_classes}")
print(f"Number of tasks: {scenario_val.nb_tasks}")
exemplar_set = ExemplarSet(transform=train_transform)
criterion_cls = nn.CrossEntropyLoss()
previous_net = None
accs = []
for task_id, train_taskset in enumerate(scenario_train):
    print(f"TASK {task_id}")
    val_taskset = scenario_val[:task_id+1]
    val_loader = DataLoader(val_taskset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_lr = MultiStepLR(optimizer, milestones=scheduling, gamma=args.gamma) 

    if task_id == 0 or args.exR == False:
        train_loader = DataLoader(train_taskset, batch_size=args.batch_size, shuffle=True)
        memory_loader = None
        lamda = None
    else:
        train_loader = DataLoader(train_taskset, batch_size=args.batch_size // 2, shuffle=True)
        memory_loader = DataLoader(exemplar_set, batch_size=args.batch_size // 2, shuffle=True)
        lamda = args.lamda_base * np.sqrt(train_taskset.nb_classes / scenario_train[:task_id].nb_classes)
    # best_acc_on_task = 0
    for epoch in range(args.epochs):
        train(args, train_loader, memory_loader, model, task_id, criterion_cls, previous_net, optimizer, epoch, lamda)
        acc_val, loss_val = validate(model, val_loader=val_loader)
        scheduler_lr.step()
        print(f"VALIDATION \t Epoch: {epoch}/{args.epochs}\t loss: {loss_val}\t acc: {acc_val}")
    accs.append(acc_val)
    print(f"ACCURACY \t  {acc_val}")
    if task_id < scenario_train.nb_tasks - 1:
        if args.exR:
            for c in train_taskset.get_classes():
                images_in_c, labels_in_c = get_all_images_per_class(train_taskset, c)
                indexes = perform_selection(args, images_in_c, labels_in_c, model, val_transform)
                exemplar_set.update_data_memory(images_in_c[indexes], labels_in_c[indexes])
        previous_net = copy.deepcopy(model)
        model.expand_classes(scenario_train[task_id+1].nb_classes)
        model.cuda()
print(accs)