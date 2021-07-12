import enum
import torch
import torch.nn as nn
from utils.AverageMeter import AverageMeter
from utils.utils import accuracy


def validate(net, val_loader):
    criterion = nn.CrossEntropyLoss()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.cuda()
    net.eval
    with torch.no_grad():
        for batch_id, (inputs, targets, t) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, output = net(inputs)
            acc_val = accuracy(output, targets, topk=(1,))  # forse mettere anche top5
            loss = criterion(output, targets)
            acc_meter.update(acc_val[0].item(), inputs.size(0))
            loss_meter.update(loss.item(), inputs.size(0))
    return acc_meter.avg, loss_meter.avg


def extract_features(args, net, loader):
    features = None
    net.cuda()
    net.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            # inputs = inputs.cuda(args.device)
            inputs = inputs.cuda()
            f, _ = net(inputs)
            # f = l2_norm(f)
            if features is not None:
                features = torch.cat((features, f), 0)
            else:
                features = f

    return features.detach().cpu().numpy()


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output
