# Train and Test

#### Contents
1. Image Retrieval
   1. [Split training set](#Split-training-set)
   2. [Train old model](#Train-old-model)
   3. [Train new (compatible) model](#Train-new-(compatible)-model)
2. Face Recognition



## Image Retrieval

##### Split training set

  Model upgrades usually happen when (i) training data is increased or (ii) the model structure is strengthened.
  
  To simulate the scenario (i), we first run the following script to split the whole training set by data (cmd: --split_gldv2_by_data) or class (cmd: --split_gldv2_by_class).

  ```python
  # split by data
  python -m utils.split_dataset --root_path '../annotation/gldv2' --split_gldv2_by_data --split_ratio 0.3

  # split by class
  python -m utils.split_dataset --root_path '../annotation/gldv2' --split_gldv2_by_class --split_ratio 0.3
  ```
  

##### Train old model
Download pretrained resnet models from , and save them into `./pretrained_model/`.

DistributedDataParallel (DDP) Training command:

  ```python
  python train.py --name "old_resnet50" --train_data_dir './data/gldv2/' --train_img_list './annotation/gldv2/gldv2_old_30percent_data.txt' --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --arch resnet50  --pretrained_model_path "./pretrained_model/resnet50-0676ba61.pth" --epochs 30 -bs 192 --workers 8 --lr_scheduler cosine --lr 0.1  --val_period 5 -c config.json --use_amp
  ```

| `Command`          | `Description`                                | `Options`                             |
| ------------------ | -------------------------------------------- | ------------------------------------- |
| `--arch`           | `-`                                          | `resnet50 / resnet101 / inception_v3` |
| `--lr_scheduler`   | `-`                                          | `cosine / step`                       |
| `--train_data_dir` | `image path`                                 | `-`                                   |
| `--train_img_list` | `index file`                                 | `-`                                   |
| `--use_amp`        | `turn on automatic mixed precision training` | `-`                                   |


##### Train new (compatible) model
DistributedDataParallel (DDP) Training command:

  ```python
  python train_bct.py --name "new_resnet50" --train_data_dir './data/gldv2/' --train_img_list './annotation/gldv2/label_81313.txt' --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --arch resnet50  --pretrained_model_path "./pretrained_model/resnet50-0676ba61.pth" --epochs 30 -bs 192 --workers 8 --lr_scheduler cosine --lr 0.1  --val_period 5 -c config.json --use_amp --old_arch resnet50 --old_pretrained_model_path './saved/old_resnet50/xxx/models/ckpt/xxx.pth.tar' --comp_loss_type contra_ract --comp_loss_topk 10
  ```

| `Command`                    | `Description`                     | `Options`                                                    |
| ---------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `--comp_loss_type`           | `-`                               | bct / lfw / fd / l2 / contra / triplet / bct_ract / contra_ract / triplet_ract |
| `----comp_loss_topk`         | select top-k negative samples     | -                                                            |
| `--comp_loss_temp`           | `temperature in contrastive loss` | `-`                                                          |
| `--comp_loss_triplet_margin` | `margin in triplet loss`          | `-`                                                          |
| `--comp_loss_weight`         | `compatible loss weight`          | `-`                                                          |