from os import W_OK
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
from utils.AverageMeter import AverageMeter
from utils.utils import extract_batch_from_memory, accuracy, save_model
from loss.less_forget import EmbeddingsSimilarity
from loss.margin_lucir import ucir_ranking


def train(args, loader_train, net, task_id, criterion_cls, previous_net, optimizer, epoch, lamda, old_classes):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    net.train()
    for batch_id, (inputs, targets, t) in enumerate(loader_train):
        inputs, targets = inputs.cuda(), targets.cuda()
        feature, output = net(inputs)
        loss = criterion_cls(output, targets)
        if task_id > 0 and previous_net is not None:
            with torch.no_grad():
                feature_old, output_old = previous_net(inputs)
            if args.less_forg:
                loss_less_forget = EmbeddingsSimilarity(l2_norm(feature_old), l2_norm(feature))
                loss += lamda * loss_less_forget
            if args.ranking:
                mask =[False if i in old_classes else True for i in targets]
                #index = torch.where(targets == x)
                loss_margin = ucir_ranking(logits=output[mask],
                                        targets=targets[mask],
                                        task_size=args.increment,
                                        nb_negatives=args.knegatives,
                                        margin=args.ranking
                                        )
                loss += loss_margin
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_training = accuracy(output, targets, topk=(1,))
        acc_meter.update(acc_training[0].item(), inputs.size(0))
        loss_meter.update(loss.item(), inputs.size(0))
    
    print(f"TRAIN \t Epoch: {epoch}/{args.epochs}\t loss: {loss_meter.avg}\t acc: {acc_meter.avg}")



def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output