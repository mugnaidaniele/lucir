import torch
from torch._C import dtype
import torch.nn as nn

def ucir_ranking(logits, targets, task_size, nb_negatives=2, margin=0.2):
    """Hinge loss from UCIR.

    Taken from: https://github.com/hshustc/CVPR19_Incremental_Learning

    # References:
        * Learning a Unified Classifier Incrementally via Rebalancing
          Hou et al.
          CVPR 2019
    """
    gt_index = torch.zeros(logits.size()).to(logits.device)
    gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5) # One-hot encoding

    gt_scores = logits.masked_select(gt_index)  # get top-K scores on novel classes

    num_old_classes = logits.shape[1] - task_size
    max_novel_scores = logits[:, num_old_classes:].topk(nb_negatives, dim=1)[0]  # the index of hard samples, i.e., samples of old classes
    hard_index = targets.lt(num_old_classes)
    hard_num = torch.nonzero(hard_index).size(0)

    if hard_num > 0:
        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, nb_negatives)
        max_novel_scores = max_novel_scores[hard_index]
        assert (gt_scores.size() == max_novel_scores.size())
        assert (gt_scores.size(0) == hard_num)
        loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1), \
            max_novel_scores.view(-1, 1), torch.ones(hard_num*nb_negatives).to(logits.device))
        return loss

    return torch.tensor(0).float()