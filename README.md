[![pytorch](https://img.shields.io/badge/pytorch-1.10.0-%2523ee4c2c.svg)](https://pytorch.org/)


# Temporal Scene Montage for Self-Supervised Video Scene Boundary Detection

This is an official PyTorch Implementation of **Temporal Scene Montage for Self-Supervised Video Scene Boundary Detection**.

<!-- <p align="center"><img width="100%" src="model.jpg"></p> -->

## Environment

This project runs on Linux (Ubuntu 22.04) with one GPU (~4G) and a large memory (~80G).

Install the following packages at first:
- python 3.9.2
- PyTorch 1.10.0
- torchvision 0.11.1
- torchmetrics 0.9.3
- pandas
- munch
- h5py
- vit_pytorch
- omegaconf

## Prepare datasets

Commnads for preparing datasets can be found in `preprocess.sh`.

### Extract visual features

For perform training process faster, we save visual features in `.pkl` files. For example, the structure of `ImageNet_shot.pkl` is as the following:
``` json
{"tt0000000":
    {
        "0000":array(),
        "0001":array(),
        ···
    },
"tt0000001":
    {
        ···
    },
...
}
```
There in, `tt0000000` is a video's ID. For each video, the key `0000` indicates a shot's ID. Each shot is encoded as a feature vector of 2048-dim.

### Labels for datasets

For convinient, labels are reformated and saved into `.pkl` files, too.

- `shot_annotation.pkl` saves the indices of the first frame and the last frame for each shot.
- `scene_annotation.pkl` saves the indices of the first shot and the last shot for each scene.
- `label_dict.pkl` saves a list for each video, where each element in the list indicates whether a shot is the first shot of a scene or not.

Codes for generating the above files can be found in `preprocess.ipynb`.


## Train & Test

Commands for train&test can be seen in `runner.sh`. Some ablations can be seen in `runner2.sh` and `runner3.sh`. Here we show some basic commands.

Pre-training:
``` bash
python -m src.pretrain config/selfsup_best.yaml
```

Fine-tuning:

``` bash
python -m src.finetune config/selfsup_best.yaml
```

Test:

``` bash
python -m src.evaluate config/selfsup_best.yaml
```

## Visualization

- Codes for plotting data points can be seen in `show_log.ipynb`.
- Codes for drawing heatmaps can be seen in `visualize.ipynb`.

## Parameters

Configuration files `config/xxx.yaml` contains all the hyperparameters.
- `base` contains basic configuration. Among them, 
    - `base.params.clip_len` indicates the number in each clip.
    - `base.path` includes file paths of formatted datasets and labels.
    - `model` is the basename of the Model code file.
- `pretrain`, `finetune` and `evaluate` correspond to two training stages and the testing stage.
    - `pretrain.params.label_percentage` specifies the percentage of data to use during pre-training.
    - `finetune.aim_index` specifies the index of films in OVSD/BBC for evaluation using leave-one-out method.
    - `finetune.load_path` specifies the path of the pre-trained model.
    - `finetune.train` is the basename of the Dataset code file.
    - `finetune.vid_list` indicates which subset of MovieNet to use.
    - `evaluate.head` specifies the prediction header to use.