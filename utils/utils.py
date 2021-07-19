import torch
import numpy as np
from torchvision import datasets, models, transforms
from continuum.datasets import CIFAR100
from torch.utils.data import WeightedRandomSampler

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def next_with_clause(iterator, data_loader):
    try:
        x, y = next(iterator)
    except Exception as e:
        iterator = iter(data_loader)
        x, y = next(iterator)
    return x, y, iterator


def extract_batch_from_memory(memory_iterator, memory_data_loader, batch_size):
    """
    Extract a batch from memory
    """
    memory_imgs, memory_labels, memory_iterator = next_with_clause(memory_iterator, memory_data_loader)

    assert memory_labels.shape[0] == memory_imgs.shape[0]
    while memory_labels.shape[0] < batch_size:
        m, l, memory_iterator = next_with_clause(memory_iterator, memory_data_loader)
        memory_imgs = torch.cat((memory_imgs, m))
        memory_labels = torch.cat((memory_labels, l))
        memory_iterator = iter(memory_data_loader)

    memory_imgs = memory_imgs[:batch_size]
    memory_labels = memory_labels[:batch_size]

    return memory_imgs, memory_labels, memory_iterator


def get_all_images_per_class(task_set, target):
    indexes = np.arange(len(task_set))
    images, labels, task = task_set.get_raw_samples(indexes)

    images_of_class = images[labels == target]
    labels_of_class = labels[labels == target]

    return images_of_class, labels_of_class

def getTransform(dataset):
    if dataset == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])
        val_transform =transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])
        test_transform =transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])

    elif dataset == "cifar100":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        train_transform =[
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
                        ]
        val_transform =[
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
        ]
        test_transform =[
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
        ]

        pass
    else:
        raise "Dataset not recognized"
    return train_transform, val_transform, test_transform

def get_dataset(dataset):
    if dataset == "cifar100":
        dataset_train = CIFAR100("data", download=True, train=True)
        dataset_test = CIFAR100("data", download=True, train=False)
    else:
        pass
    return dataset_train, dataset_test

def get_sampler(target):
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def save_model(net, task_id):
    net_state_dict = net.state_dict()
    checkpoint ={
        'net_state_dict': net_state_dict,
                }
    ckpt_path = f"ckpt_task_{task_id}.pt" 
    torch.save(checkpoint, ckpt_path)

def load_model(net, task_id):
    ckpt_path = f"ckpt_task_{task_id}.pt"
    net.load_state_dict(torch.load(ckpt_path)['net_state_dict'])
    return net