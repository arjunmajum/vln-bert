# Improving Vision-and-Language Navigation with Image-Text Pairs from the Web

Arjun Majumdar, Ayush Shrivastava, Stefan Lee, Peter Anderson, Devi Parikh, and Dhruv Batra

Paper: https://arxiv.org/abs/2004.14973

## Model Zoo

A variety of pre-trained VLN-BERT weights can accessed through the following links:

| |Pre-training Stages|Job ID|Val Unseen SR|URL|
|-|:-------------:|:----:|:-----------:|:-:|
|0|no pre-training|174631|30.52%|TBD|
|1|1|175134|45.17%|TBD|
|3|1 and 2|221943|49.64%|[download](https://dl.dropbox.com/s/v9qmgnjrdx9dpdc/run_221943_pytorch_model_16.bin)|
|2|1 and 3|220929|50.02%|[download](https://dl.dropbox.com/s/hvp62zlsccxk54b/run_220929_pytorch_model_14.bin)|
|4|1, 2, and 3 (Full Model)|220825|59.26%|[download](https://dl.dropbox.com/s/hel0ujgn94iwh26/run_220825_pytorch_model_10.bin)|

## Usage Instructions

Follow the instructions in [INSTALL.md](INSTALL.md) to setup this codebase.
The instructions walk you through several steps including preprocessing the
Matterport3D panoramas by extracting regions with a pretrained object
detector.

### Training

To preform stage 3 of pre-training, first download ViLBERT weights from
[here](https://dl.dropbox.com/s/vjilqowlaobsxc6/vilbert_pytorch_model_9.bin).
Then, run:
```
python \
-m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
train.py \
--from_pretrained <path/to/vilbert_pytorch_model_9.bin> \
--save_name [pre_train_run_id] \
--num_epochs 50 \
--warmup_proportion 0.08 \
--cooldown_factor 8 \
--masked_language \
--masked_vision \
--no_ranking
```

To fine-tune VLN-BERT for the path selection task, run:

```
python \
-m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
train.py \
--from_pretrained <path/to/pytorch_model_50.bin> \
--save_name [fine_tune_run_id]
```


### Evaluation

To evaluate a pre-trained model, run:

```
python test.py \
--split [val_seen|val_unseen] \
--from_pretrained <path/to/run_[run_id]_pytorch_model.bin> \
--save_name [run_id]
```

followed by:

```
python scripts/calculate-metrics.py <path/to/results_[val_seen|val_unseen].json>
```

## Citation

If you find this code useful, please consider citing:

```
@inproceedings{majumdar2020improving,
  title={Improving Vision-and-Language Navigation with Image-Text Pairs from the Web},
  author={Arjun Majumdar and Ayush Shrivastava and Stefan Lee and Peter Anderson and Devi Parikh and Dhruv Batra},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
