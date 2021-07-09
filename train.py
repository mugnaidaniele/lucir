from os import W_OK
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
from utils.AverageMeter import AverageMeter
from utils.utils import extract_batch_from_memory, accuracy
from loss.less_forget import EmbeddingsSimilarity
from loss.margin_lucir import ucir_ranking


def train(args, loader_train, loader_memory, net, task_id, criterion_cls, previous_net, optimizer, epoch, lamda):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    net.train()
    memory_iterator = iter(loader_memory) if (task_id > 0 and args.exR) else None
    for batch_id, (inputs, targets, t) in enumerate(loader_train):
        if memory_iterator is not None:
            inputs_from_memory, targets_from_memory, memory_iterator = extract_batch_from_memory(memory_iterator, loader_memory, args.batch_size_train)
            inputs = torch.cat((inputs, inputs_from_memory))
            targets = torch.cat((targets, targets_from_memory))
        inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
        feature, output = net(inputs)
        loss = criterion_cls(output, targets)
    if task_id > 0 and previous_net is not None:
        with torch.no_grad():
            feature_old, output_old = previous_net(inputs)
        if args.less_forg:
            loss_less_forget = EmbeddingsSimilarity(l2_norm(feature_old), l2_norm(feature))
            loss += lamda * loss_less_forget
        if args.ranking:
            # todo output old e basta
            loss_margin = ucir_ranking(logits=output,
                                       targets=targets,
                                       task_size=args.increment,
                                       nb_negatives=max(2, args.increment),
                                       margin=0.5
                                       )
            loss += loss_margin
        if args.mimic:
            pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if batch_id + 1 % (len(loader_train) // 2) == 0:
        print(f"Epoch: {epoch}/{args.n_epochs}\t loss: {loss_meter.avg}\t acc: {acc_meter.avg}")
    acc_training = accuracy(output, targets, topk=(1,))
    acc_meter.update(acc_training[0].item(), inputs.size(0))
    loss_meter.update(loss.item(), inputs.size(0))


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output