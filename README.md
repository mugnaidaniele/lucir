# Learning a Unified Classifier Incrementally via Rebalancing
Unofficial Implementation of paper _Learning a Unified Classifier Incrementally via Rebalancing_  [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)
### Requirements
`python3 -m venv lucir-env`

`source lucir-env/bin/activate`

`pip install requirements.txt`

### Running
`python main.py --args`

### Hyper-parameters

`--dataset` Dataset [CIFAR100, IMAGENET]

`--start` Number of classes of first task

`--increment` Number of classes at each next task

`--rehearsal` Number of example stored per each class

`--selection` Selection of exemplar [Herding, Random, Closest to Mean]

`--exR` if True, Exemplar are stored

`--class_balance_finetuning` if True, a class balance fine-tuning is performed at the end of each task

`--less_forg`  if True,  _**less-forget**_ constraint is used

`--lambda_base` weight factor of less-forget loss

`--ranking` if True,  _**margin ranking loss**_ constraint is used
### Comparison with original results

#### CIFAR-100

<table>
  <tr>
    <td>Starting Classes</td>
    <td>Increment</td>
    <td colspan="2">Average Incremental Accuracy</td>
  </tr>
  <tr>
    <td>50</td>
    <td>50</td>
    <td>69.81</td>
    <td>Here</td>
  </tr>

  <tr>
    <td>50</td>
    <td>25</td>
    <td>66.76</td>
    <td>Here</td>
  </tr>

  <tr>
    <td>50</td>
    <td>10</td>
    <td>63.42</td>
    <td>Here</td>
  </tr>

  <tr>
    <td>50</td>
    <td>5</td>
    <td>60.18</td>
    <td>Here</td>
  </tr>
</table>

#### IMAGENET-100
