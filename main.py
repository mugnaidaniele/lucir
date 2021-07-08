import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from models import ResNet32Cifar
from continuum.datasets import CIFAR100
from continuum import ClassIncremental
#from train import train
from validate import validate
from utils.ExemplarSet import ExemplarSet
from utils.utils import get_all_images_per_class
from torch.optim.lr_scheduler import MultiStepLR
from utils.feature_selection import perform_selection
import os
import numpy as np
import copy
dataset = "cifar100"
start = 50
increment = 10
bs = 64
cosine = True
lr = 0
momentum = 0
scheduling = [0, 1]
n_epochs = 0
exR = True
exemplar_per_class = 20
lamda_base = 0
margin = 0
k_negatives = 0
gamma = 0
rehearsal = 20


parser = argparse.ArgumentParser(description='Learning unified classifier via rebalancing')


args = parser.parse_args()

model = ResNet32Cifar(10, cosine)
dataset_train = CIFAR100("data", download=True, train=True)
dataset_test = CIFAR100("data", download=True, train=False)

scenario_train = ClassIncremental(dataset_train, increment=increment, initial_increment=start)
scenario_val = ClassIncremental(dataset_test, increment=increment, initial_increment=start)
assert scenario_train.nb_tasks == scenario_val.nb_tasks
print(f"Number of classes: {scenario_train.nb_classes}")
print(f"Number of tasks: {scenario_val.nb_tasks}")

exemplar_set = ExemplarSet()
criterion_cls = nn.CrossEntropyLoss()
previous_net = None

for task_id, train_taskset in enumerate(scenario_train):
    print(task_id)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # scheduler_lr = MultiStepLR(optimizer, milestones=scheduling, gamma=gamma)
    # lamda = lamda_base * np.sqrt(train_taskset.nb_classes / scenario_train[:task_id].nb_classes)
    # train_loader = DataLoader(train_taskset, batch_size=bs, shuffle=True)
    # memory_loader = DataLoader(exemplar_set)
    # best_acc_on_task = 0
    # for epoch in range(args.n_epochs):
    #     train(args, train_loader, memory_loader, model, task_id, criterion_cls, previous_net, optimizer,epoch)
    #     acc_val = validate()
    #     if acc_val > best_acc_on_task:
    #         best_acc_on_task = acc_val
    
    # if args.exR > 0:
    #     for c in train_taskset.get_classes:
    #         images_in_c, labels_in_c = get_all_images_per_class(train_taskset, c)
    #         indexes = perform_selection(args, images_in_c, labels_in_c, model, val_transform)
    #         memory_dataset.update_data_memory(images_in_c[indexes], labels_in_c[indexes])
    if task_id < scenario_train.nb_tasks-1:
        for c in train_taskset.get_classes:
            images_in_c, labels_in_c = get_all_images_per_class(train_taskset, c)
            indexes = perform_selection(args, images_in_c, labels_in_c, model, val_transform)
            exemplar_set.update_data_memory(images_in_c[indexes], labels_in_c[indexes])
    #     previous_net = copy.deepcopy(model)
    #     model.expand_classes(scenario_train.nb_classes)



