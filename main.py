import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from models import create
from continuum.datasets import CIFAR100
from continuum import ClassIncremental
import models
from train import train, balanced_finetuning
from validate import validate
from utils.ExemplarSet import ExemplarSet
from utils.utils import get_all_images_per_class
from torch.optim.lr_scheduler import MultiStepLR
from utils.feature_selection import perform_selection
from utils.utils import getTransform, get_dataset, get_sampler, save_model, load_model
from torchvision.transforms import transforms

import os
import numpy as np
import copy


parser = argparse.ArgumentParser(description='Learning unified classifier via rebalancing')
parser.add_argument('--dataset', type=str, default="cifar100", metavar='BATCH', help='dataset')
parser.add_argument('--start', type=int, default=50, help='starting classes')
parser.add_argument('--increment', type=int, default=50, help='increment classes at each task')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for cnn')
parser.add_argument('--momentum', type=float, default=0.9 , help='momentum')
parser.add_argument('--epochs', type=int, default=160,  help='number of epochs')
parser.add_argument('--lamda_base', type=float, default=5.0,help='weight factor for feat dist')
parser.add_argument('--margin', type=float, default=0.5, help='margin of loss margin')
parser.add_argument('--gamma', type=float, default=0.1, help='factor for multiply learning rate')
parser.add_argument('--knegatives', type=int, default=2, help='k negatives for loss margin')
parser.add_argument('--rehearsal', type=int, default=20,  help='examplar stored per class')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='weight decay')
parser.add_argument('--selection', type=str, default="herding",  help='type of selection of exemplar')
parser.add_argument("--exR", action="store_true", default=True, help="experience replay")
parser.add_argument("--cosine", action="store_true", default=True, help="cosine classifier")
parser.add_argument("--class_balance_finetuning", action="store_true", default=False, help="class_balance_finetuning ")
parser.add_argument('--ft_epochs', default=20, type=int, help='Epochs for class balance finetune')
parser.add_argument('--ft_base_lr', default=0.01, type=float,help='Base learning rate for class balance finetune')
parser.add_argument('--ft_lr_strat', default=10, type=int, nargs='+', help='Lr_strat for class balance finetune')
parser.add_argument("--less_forg", action="store_true", default=True, help="less forgetting loss")
parser.add_argument("--ranking", action="store_true", default=True, help="loss margin ranking")

parser.add_argument("--list", nargs="+", default=["80", "120"])

args = parser.parse_args()
#print(args.ft_lr_strat)
#print([int(args.ft_lr_strat)])
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
scheduling = [int(args.list[0]) , int(args.list[1])]

criterion_cls = nn.CrossEntropyLoss()
previous_net = None
accs = []
task_classes = []
for task_id, train_taskset in enumerate(scenario_train):
    task_classes.extend(train_taskset.get_classes())
    if task_id > 0:
        train_taskset.add_samples(exemplar_set.data, exemplar_set.targets)
    print(f"TASK {task_id}")
    val_taskset = scenario_val[:task_id+1]
    sampler = get_sampler(train_taskset._y)
    val_loader = DataLoader(val_taskset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_lr = MultiStepLR(optimizer, milestones=scheduling, gamma=args.gamma) 
    train_loader = DataLoader(train_taskset, batch_size=args.batch_size, sampler=sampler)
    if task_id == 0:
        lamda = None
        old_classes = None
    else:
        lamda = args.lamda_base * np.sqrt(scenario_train[task_id].nb_classes / scenario_train[:task_id].nb_classes)
        old_classes = scenario_train[:task_id].get_classes()
    best_acc_on_task = 0
    for epoch in range(args.epochs):
        train(args, train_loader, model, task_id, criterion_cls, previous_net, optimizer, epoch, lamda, old_classes)
        acc_val, loss_val = validate(model, val_loader)
        if acc_val > best_acc_on_task:
            best_acc_on_task = acc_val
        #     print(f"Saving best model\t ACC:{best_acc_on_task}")
        #     save_model(model, task_id)
        scheduler_lr.step()
        print(f"VALIDATION \t Epoch: {epoch}/{args.epochs}\t loss: {loss_val}\t acc: {acc_val}")
    # load best ckpt
    #print("Loading best model...")
    #model = load_model(model, task_id)
    #model.cuda()

    #if task_id < scenario_train.nb_tasks - 1:
    if args.exR:
        print(f"Selecting {args.rehearsal} exemplar per class from task {task_id}")
        for c in task_classes:
            images_in_c, labels_in_c = get_all_images_per_class(train_taskset, c)
            indexes = perform_selection(args, images_in_c, labels_in_c, model, val_transform)
            exemplar_set.update_data_memory(images_in_c[indexes], labels_in_c[indexes])
        
    if args.class_balance_finetuning and task_id > 0:
        print("Class Balance Finetuning")
        optimizer = optim.SGD(model.parameters(), lr=args.ft_base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler_lr = MultiStepLR(optimizer, milestones=[int(args.ft_lr_strat)], gamma=args.gamma)
        loader_balanced = DataLoader(exemplar_set, batch_size=args.batch_size, shuffle=True)
        best_acc_on_task = 0
        for epoch in range(args.ft_epochs):
            balanced_finetuning(args, loader_balanced, model, task_id, criterion_cls, optimizer,epoch)
            acc_val, loss_val = validate(model, val_loader)
            print(f"VALIDATION \t Epoch: {epoch}/{args.ft_epochs}\t loss: {loss_val}\t acc: {acc_val}")
            if acc_val > best_acc_on_task:
                best_acc_on_task = acc_val
            scheduler_lr.step()
        #print(f"VALIDATION \t Epoch: {epoch}/{args.epochs}\t loss: {loss_val}\t acc: {acc_val}")
        accs.append(best_acc_on_task)
        #print(f"ACCURACY \t  {acc_val}")
    else:
        accs.append(best_acc_on_task)
    if task_id < scenario_train.nb_tasks - 1 :  
        #print("Expanding")
        previous_net = copy.deepcopy(model)         
        model.expand_classes(scenario_train[task_id+1].nb_classes)
        model.cuda()
        task_classes = []

print(accs)
print(np.mean(np.array(accs)))