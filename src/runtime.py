import torch, os
from .common.warmup_lr import warmup_decay_cosine
from .common.config import *
from .common.collate_fn import collate_fn
from importlib import import_module


# data_loader
pin_memory = cfg.base.get('DEVICE', 'cpu') != 'cpu'
stage = cfg.base.stage
conf = cfg[stage]
if conf.get('train'):
    TrainDataset = getattr(import_module(f'src.dataset.{conf.train.dataset}'), 'Dataset')
    train_loader = torch.utils.data.DataLoader(
        dataset=TrainDataset(data[conf.train.vid_list]),
        batch_size=conf.params.batch_size,
        shuffle=True,
        num_workers=conf.params.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=pin_memory
    )
if conf.get('test'):
    TestDataset = getattr(import_module(f'src.dataset.{conf.test.dataset}'), 'Dataset')
    val_loader = torch.utils.data.DataLoader(
        dataset=TestDataset(data[conf.test.vid_list]),
        batch_size=conf.params.batch_size,
        shuffle=False,
        num_workers=conf.params.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory
    )

# model
Model = getattr(import_module(f'src.model.{cfg.base.model}'), 'Model')
model = Model().to(cfg.base.DEVICE)
if cfg[stage].get('load_path'):
    if cfg[stage].get('load_path') != 'None':
        print(stage, cfg[stage].get('load_path'))
        state_dict = torch.load(conf.load_path, 'cpu')
        model.load_state_dict(state_dict['parameters'])

# optimizer
if cfg[stage].get('optim'):

    optimizer = eval('torch.optim.'+conf.optim)(
        model.parameters(),
        lr=conf.params.lr
    )

# lr_scheduler
if stage not in ['evaluate', 'extract']:
    iter_num = len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_decay_cosine(iter_num, iter_num*(conf.params.total_epochs-1))
    )