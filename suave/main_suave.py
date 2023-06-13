# Based on https://github.com/facebookresearch/swav/blob/main/main_swav.py

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

import argparse
import math
import os
import shutil
import time
from logging import getLogger
import urllib
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
    cycle,
    cosine_scheduler,
    get_module,
    LARS,
)
from src.multicropdataset import MultiCropDataset
from src.mixup import MultiViewCollateMixup, CollateMixup, collate_rep_aug
import src.resnet50 as resnet_models
from src.augment import Augment3

logger = getLogger()

def get_args_parser():
    parser = argparse.ArgumentParser("Suave", add_help=False)

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                        help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")
    parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
                        help="fine-tune on either 1% or 10% of labels")
    parser.add_argument("--mixup_unlab", type=bool_flag, default=False,
                        help="whether to use mixup on unlabeled images")
    parser.add_argument("--mixup_unlab_type", type=str, default="all", choices=["all", "globals"],
                        help="whether to use mixup on all views or on global views only")
    parser.add_argument("--mixup_lab", type=bool_flag, default=False,
                        help="whether to use mixup on labeled images")
    parser.add_argument("--mixup_lab_type", type=str, default="extend", choices=["extend", "replace"],
                        help="whether to extend or replace the batch with mixed imgs")
    parser.add_argument("--mixup_lab_prob", type=float, default=0.5,
                        help="probability of applying mixup when mixup replace is used")
    parser.add_argument("--augment3_lab", type=bool_flag, default=False,
                        help="whether to use 3Augment (from deit III paper)")
    parser.add_argument("--simple_crop", type=bool_flag, default=False,
                        help="whether to use simple resized crop within 3Augment")
    parser.add_argument("--light_augment_lab", type=bool_flag, default=False,
                        help="whether to use lighter augmentation on labeled examples")
    parser.add_argument("--color_jit_unlab", type=bool_flag, default=True,
                        help="whether to use lighter augmentation on unlabeled examples")

    #########################
    ## swav specific params #
    #########################
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                        help="list of crops id used for computing assignments")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="temperature parameter in training loss")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--feat_dim", default=128, type=int,
                        help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=1000, type=int,
                        help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=0,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=15,
                        help="from this epoch, we start using a queue")
    parser.add_argument("--label_smoothing", type=float, default=0.01,
                        help="label smoothing for supervised loss")

    #########################
    ### other model params ##
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--hidden_mlp", default=[2048], type=int, nargs="+",
                        help="hidden layer dimension in projection head")
    parser.add_argument("--multihead_eval", type=bool_flag, default=False,
                        help="whether to online train one classifier for each projection layer")
    parser.add_argument("--ensemble_eval", type=bool_flag, default=False,
                        help="whether to train a linear classifier on top of features ensemble(concat)")
    parser.add_argument("--bn_ensemble", type=bool_flag, default=False,
                        help="whether to use batch norm prior to the ensemble linear layer")
    parser.add_argument("--detach_ensemble", type=bool_flag, default=True,
                        help="whether to compute ensemble layer on a detached branch")
    parser.add_argument('--use_momentum', default=False, type=bool_flag, 
                        help="flag for enabling momentum encoder")                 
    parser.add_argument('--momentum_encoder', default=0.996, type=float,
                        help="""Base EMA parameter for ema update of the encoder. 
                        The value is increased to 1 during training with cosine schedule.
                        We recommend setting a higher value with small batches:
                        for example use 0.9995 with batch size of 256.""")

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--lab_batch_size", default=50, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=0, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")
    parser.add_argument("--warmup_proj_nepochs", default=2, type=int,
                        help="freeze the backbone to allow the warmup of the projection head")
    parser.add_argument("--init_proj", type=bool_flag, default=False,
                        help="whether to initialize the projection head from swav pretraining")
    parser.add_argument("--swav_init_ckpt", type=str, default="swav_800ep_pretrain.pth.tar",
                        help="Checkpoint used to iniziatile SwAV")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int, help="""
                        number of processes: it is set automatically and
                        should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=25,
                        help="Save the model periodically")
    parser.add_argument("--use_fp16", type=bool_flag, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")
    parser.add_argument("--wandb", type=bool_flag, default=False, help="whether to use wandb or not")

    return parser


def train_suave(args):
    init_distributed_mode(args)
    fix_random_seeds(args.seed)

    train_multihead_metrics = []
    val_multihead_metrics = []
    if args.multihead_eval:
        train_multihead_metrics = (
            ["train_lab_bbone_acc1"] + \
            [f"train_lab_proj_{i}_acc1" for i in range(len(args.hidden_mlp))] + \
            ["train_lab_proj_last_acc1"] + \
            ["train_lab_ensemble_acc1"]
        )
        val_multihead_metrics = (
            ["val_bbone_acc1"] + \
            [f"val_proj_{i}_acc1" for i in range(len(args.hidden_mlp))] + \
            ["val_proj_last_acc1"] + \
            ["val_ensemble_acc1"]
        )

    logger, training_stats = initialize_exp(
        args,
        "epoch",
        "train_loss",
        "train_unlab_loss",
        "train_lab_loss",
        "train_unlab_acc1",
        "train_unlab_acc5",
        "train_lab_acc1",
        "train_lab_acc5",
        *train_multihead_metrics,
        "val_loss",
        "val_acc1",
        "val_acc5",
        *val_multihead_metrics,
    )

    # init wandb
    if args.wandb and args.rank == 0:
        wandb_id = os.path.split(args.output_dir)[-1]
        logger.info(f"WANDB active ID: {wandb_id}")
        name = wandb_id.split("-", 1)[-1]
        wandb.init(
            project="simple-semi-self",
            id=wandb_id,
            config=args,
            resume="allow",
            name=name,
            save_code=True,
        )

    train_data_path = os.path.join(args.data_path, "train")
    val_data_path = os.path.join(args.data_path, "val")

    # ============ unlabeled training data ... ============
    train_dataset = MultiCropDataset(
        train_data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        return_label=True,
        apply_color_tr=args.color_jit_unlab,
    )

    if args.mixup_unlab:
        collate_fn = MultiViewCollateMixup(
            mixup_alpha=1.,
            cutmix_alpha=1.,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            correct_lam=True,
            label_smoothing=0.0,
            num_classes=args.nmb_prototypes,
            views_to_mix=args.mixup_unlab_type
        )
    else:
        collate_fn = None
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # ============ build labeled data ... ============
    lab_dataset = datasets.ImageFolder(train_data_path)
    # take either 1% or 10% of images
    subset_file = open(f"subsets/{args.labels_perc}percent.txt").readlines()
    list_imgs = [li.strip() for li in subset_file]
    list_img_classes = []
    for i in range(len(list_imgs)):
        list_img_classes.append([
            k for k in lab_dataset.class_to_idx.keys()
            if k.startswith(list_imgs[i].split('_')[0])
        ][0])
    lab_dataset.samples = [(
        os.path.join(train_data_path, lic, li),
        lab_dataset.class_to_idx[lic]
    ) for lic, li in zip(list_img_classes, list_imgs)]
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )

    tr_lab = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        tr_normalize,
    ])
    if args.augment3_lab:
        tr_lab = Augment3(
            224,
            data_mean=[0.485, 0.456, 0.406],
            data_std=[0.228, 0.224, 0.225],
            simple_crop=args.simple_crop,
            jitter=0.3,
            n_rep=2,
        )
    if args.light_augment_lab:
        tr_lab = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.8),
            transforms.ToTensor(),
            tr_normalize,
        ])

    lab_dataset.transform = tr_lab
    if args.mixup_lab:
        collate_fn = CollateMixup(
            extend_batch=True if args.mixup_lab_type == "extend" else False,
            mixup_alpha=1.,
            cutmix_alpha=1.,
            cutmix_minmax=None,
            prob=1.0 if args.mixup_lab_type == "extend" else args.mixup_lab_prob,
            switch_prob=0.5,
            mode='batch',
            correct_lam=True,
            label_smoothing=0.0,
            num_classes=args.nmb_prototypes,
        )
    else:
        collate_fn = collate_rep_aug
    lab_sampler = torch.utils.data.distributed.DistributedSampler(lab_dataset)
    lab_loader = torch.utils.data.DataLoader(
        lab_dataset,
        sampler=lab_sampler,
        batch_size=args.lab_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    logger.info("Building labeled data done with {} images loaded.".format(len(lab_dataset)))
    # lab_iterable = (x for _ in count() for x in lab_loader)
    lab_iterable = cycle(lab_loader)


    # ============ build labeled data ... ============
    val_dataset = datasets.ImageFolder(val_data_path)
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    # ============ build model ============
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
        online_eval_multi=args.multihead_eval,
        ensemble_eval=args.ensemble_eval,
        bn_ensemble=args.bn_ensemble,
        detach_ensemble=args.detach_ensemble,
    )
    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = LARS(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
        eta=0.001,
    )
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # start from pretrained swav
    swav_ckpt = torch.load(args.swav_init_ckpt, map_location="cpu")
    if "state_dict" in swav_ckpt:
        swav_ckpt = swav_ckpt["state_dict"]
    del swav_ckpt["module.prototypes.weight"]   # skip prototypes
    if not args.init_proj:  # skip projection head
        proj_names = [name for name in swav_ckpt.keys() if "module.projection_head" in name]
        for name in proj_names:
            del swav_ckpt[name]
    ckpt_loaded = model.load_state_dict(swav_ckpt, strict=False)
    logger.info(ckpt_loaded)
    del swav_ckpt   # free up memory

    # momentum encoder (teacher network)
    if args.use_momentum:
        ema_model = resnet_models.__dict__[args.arch](
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
        )
        ema_model = ema_model.cuda()
        ema_model = nn.SyncBatchNorm.convert_sync_batchnorm(ema_model)
        # wrap ema_model
        ema_model = nn.parallel.DistributedDataParallel(
            ema_model,
            device_ids=[args.gpu_to_work_on]
        )

        # initialize momentum encoder to same parameters of the fast encoder 
        get_module(ema_model).load_state_dict(
            get_module(model).state_dict(), strict=False
        )
        # stop grad on the momentum encoder
        for p in ema_model.parameters():
            p.requires_grad = False

        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = cosine_scheduler(args.momentum_encoder, 1, args.epochs, len(train_loader))

        logger.info("Momentum encoder built and initialized")
    else:
        ema_model = momentum_schedule = None

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        ema_state_dict=ema_model,
        optimizer=optimizer,
        scaler=scaler if args.use_fp16 else None
    )
    start_epoch = to_restore["epoch"]

    # build the queue
    queue = None
    queue_path = os.path.join(args.output_dir, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    validate(val_loader, model)  # debug validation

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, lab_iterable, model, ema_model, momentum_schedule, optimizer, scaler, epoch, lr_schedule, queue)
        scores_val = validate(val_loader, model)
        training_stats.update(scores + scores_val)

        # save checkpoints
        if args.rank == 0:
            if args.wandb:
                # get last row of pd dataframe and trasform into a dict {col1: v1, col2: v2}
                wandb.log(training_stats.stats.iloc[-1:].to_dict("records")[-1])

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict() if args.use_momentum else None,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if args.use_fp16 else None,
            }
            torch.save(
                save_dict,
                os.path.join(args.output_dir, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.output_dir, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)

    if args.wandb and args.rank == 0:
        wandb.finish()


def train(unlab_loader, lab_iterable, model, ema_model, momentum_schedule, optimizer, scaler, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses, unlab_losses, lab_losses = AverageMeter(), AverageMeter(), AverageMeter()
    mixup_unlab_losses, online_losses = AverageMeter(), AverageMeter()
    unlab_acc, unlab_top1, unlab_top5 = [], AverageMeter(), AverageMeter()
    lab_acc, lab_top1, lab_top5 = [], AverageMeter(), AverageMeter()
    online_acc, online_top1 = [], [AverageMeter() for _ in range(len(args.hidden_mlp) + 2)]
    ensemble_acc, ensemble_top1 = [], AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, (unlab_labels, *inputs) in enumerate(unlab_loader):

        mixup_unlab_lams = None
        if args.mixup_unlab:
            mixup_unlab_lams = inputs[1]
            mixup_unlab_idxs = inputs[2]

        inputs = inputs[0]
        unlab_labels = unlab_labels.cuda(non_blocking=True)

        lab_inputs, lab_labels = next(lab_iterable)
        lab_inputs = lab_inputs.cuda(non_blocking=True)
        lab_labels = lab_labels.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(unlab_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        with torch.cuda.amp.autocast(enabled=args.use_fp16):

            # ============ normalize prototypes ... ============
            with torch.no_grad():
                w = get_module(model).prototypes.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                get_module(model).prototypes.weight.copy_(w)

                if args.use_momentum:
                    ema_w = get_module(ema_model).prototypes.weight.data.clone()
                    ema_w = F.normalize(ema_w, dim=1, p=2)
                    get_module(ema_model).prototypes.weight.copy_(ema_w)

            # ============ multi-res forward passes ... ============
            embedding, output, online_logits_multi = model([lab_inputs] + inputs)
            lab_bs, bs = lab_inputs.size(0), inputs[0].size(0)
            lab_output, unlab_output = output[:lab_bs], output[lab_bs:]
            embedding = embedding[lab_bs:].detach()

            mixed_unlab_bs = 0
            if args.mixup_unlab:
                unlab_output = unlab_output.view(-1, bs, unlab_output.size(-1))
                unlab_idxs = [i for i in range(len(unlab_output)) if i not in mixup_unlab_idxs]

                mixed_unlab_output = unlab_output[mixup_unlab_idxs]
                unlab_output = unlab_output[unlab_idxs].view(-1, unlab_output.size(-1))
                embedding = embedding.view(-1, bs, embedding.size(-1))[unlab_idxs].view(-1, embedding.size(-1))

                mixed_unlab_bs = mixed_unlab_output.size(0)
            output_for_sk  = unlab_output
            model_for_sk = model

            if args.use_momentum:
                # forward only unlabeled global views through mom enc
                with torch.no_grad():
                    embedding, output_for_sk, _ = ema_model(inputs[:args.nmb_crops[0]])
                model_for_sk = ema_model

            # ============ swav loss ... ============
            unlab_loss = 0
            unlab_acc1 = unlab_acc5 = 0
            pseudo_unlab_labels = []
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = output_for_sk[bs * crop_id: bs * (crop_id + 1)].detach()

                    # use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                get_module(model_for_sk).prototypes.weight.t()
                            ), out))
                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                    # get assignments
                    # TODO: check if apply a dynamic sharpening (temperature) as in MSN or DINO 
                    q = distributed_sinkhorn(out)[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    x = unlab_output[bs * v: bs * (v + 1)] / args.temperature
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                unlab_loss += subloss / (np.sum(args.nmb_crops) - 1)
                tmp_acc1, tmp_acc5 = accuracy(q, unlab_labels, topk=(1, 5))
                unlab_acc1 += tmp_acc1
                unlab_acc5 += tmp_acc5
                pseudo_unlab_labels.append(q)

            unlab_loss /= len(args.crops_for_assign)
            unlab_acc1 /= len(args.crops_for_assign)
            unlab_acc5 /= len(args.crops_for_assign)

            mixed_unlab_loss = 0
            if args.mixup_unlab:
                subloss = 0
                for i, lam in enumerate(mixup_unlab_lams):
                    pred = F.log_softmax(mixed_unlab_output[i] / args.temperature, dim=1)
                    subloss -= torch.mean(torch.sum(
                        lam * pseudo_unlab_labels[0] * pred + (1 - lam) * pseudo_unlab_labels[1].flip(0) * pred,
                        dim=1
                    ))
                mixed_unlab_loss += subloss / len(mixup_unlab_lams)

        # ============ supervised loss ... ============
        lab_loss = F.cross_entropy(lab_output / args.temperature, lab_labels, label_smoothing=args.label_smoothing)
        lab_acc1, lab_acc5 = accuracy(lab_output, lab_labels, topk=(1, 5))

        # ============ online eval ... ============
        online_lab_acc1 = []
        online_loss = 0
        lab_bs_no_mixed = lab_bs // 2 if args.mixup_lab and args.mixup_lab_type == "extend" else lab_bs

        if args.ensemble_eval and args.detach_ensemble:
            # esemble logits are the last element of prev_layers
            ensemble_output = online_logits_multi[-1][:lab_bs_no_mixed]
            online_logits_multi = online_logits_multi[:-1]
            online_loss += F.cross_entropy(
                ensemble_output, lab_labels[:lab_bs_no_mixed], label_smoothing=args.label_smoothing
            )
        else:
            ensemble_output = sum([
                F.softmax(logits[:lab_bs_no_mixed], dim=1)
                for logits in online_logits_multi + [lab_output / args.temperature]
            ])
        ensemble_acc1 = accuracy(ensemble_output, lab_labels[:lab_bs_no_mixed], topk=(1,))[0]
        
        for logits in online_logits_multi:
            online_loss += F.cross_entropy(
                logits[:lab_bs_no_mixed], lab_labels[:lab_bs_no_mixed], label_smoothing=args.label_smoothing
            )
            online_lab_acc1.append(accuracy(logits[:lab_bs_no_mixed], lab_labels[:lab_bs_no_mixed], topk=(1,))[0])

        # ============ total loss ... ============
        total_bs = (bs * sum(args.nmb_crops) + lab_bs + mixed_unlab_bs) 
        lab_weight = lab_bs / total_bs
        unlab_weight = bs * sum(args.nmb_crops) / total_bs
        mixed_unlab_weight = mixed_unlab_bs / total_bs
        online_eval_weight = lab_bs_no_mixed / total_bs
        loss = (
            unlab_weight * unlab_loss + \
            lab_weight * lab_loss + \
            mixed_unlab_weight * mixed_unlab_loss + \
            online_eval_weight * online_loss
        )

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # warmup proj and proto
        if epoch < args.warmup_proj_nepochs:
            for name, p in model.named_parameters():
                if "projection_head" not in name and "prototypes" not in name:
                    p.grad = None

        if args.use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if args.use_momentum:
            # EMA update for the momentum encoder
            with torch.no_grad():
                m = momentum_schedule[iteration]  # momentum parameter
                for param_name, param_k in get_module(ema_model).named_parameters():
                    param_q = get_module(model).state_dict()[param_name]
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # ============ misc ... ============
        losses.update(loss.item(), bs)
        lab_losses.update(lab_loss.item(), lab_bs)
        unlab_losses.update(unlab_loss.item(), bs)
        mixup_unlab_losses.update(mixed_unlab_loss, bs)
        online_losses.update(online_loss.item(), lab_bs)

        unlab_top1.update(unlab_acc1, bs)
        unlab_top5.update(unlab_acc5, bs)

        lab_top1.update(lab_acc1, lab_bs)
        lab_top5.update(lab_acc5, lab_bs)

        for i, top1 in enumerate(online_top1):
            top1.update(online_lab_acc1[i], lab_bs_no_mixed)
        ensemble_top1.update(ensemble_acc1, lab_bs_no_mixed)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "L {lab_loss.val:.4f} ({loss.avg:.4f})\t"
                "U {unlab_loss.val:.4f} ({unlab_loss.avg:.4f})\t"
                "M_U {mixup_unlab_loss.val:.4f} ({mixup_unlab_loss.avg:.4f})\t"
                "O_L {online_loss.val:.4f} ({online_loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lab_loss=lab_losses,
                    unlab_loss=unlab_losses,
                    mixup_unlab_loss=mixup_unlab_losses,
                    online_loss=online_losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

    unlab_acc = [unlab_top1.avg, unlab_top5.avg]
    lab_acc = [lab_top1.avg, lab_top5.avg]
    online_acc = [top1.avg for top1 in online_top1] + [ensemble_top1.avg]

    all_losses = [losses.avg, unlab_losses.avg, lab_losses.avg]
    all_acc = unlab_acc + lab_acc + online_acc
    return (epoch, *all_losses, *all_acc), queue


def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    prev_layers_top1 = [AverageMeter() for _ in range(len(args.hidden_mlp) + 2)]
    ensemble_top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for inp, target in val_loader:

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            _, output, prev_layers_output = model(inp)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1, inp.size(0))
            top5.update(acc5, inp.size(0))

            if args.ensemble_eval and args.detach_ensemble:
                # esemble logits are the last element of prev_layers
                ensemble_output = prev_layers_output[-1]
                prev_layers_output = prev_layers_output[:-1]
            else:
                ensemble_output = sum([
                    F.softmax(logits, dim=1)
                    for logits in prev_layers_output + [output]
                ])
            acc1 = accuracy(ensemble_output, target, topk=(1,))[0]
            ensemble_top1.update(acc1, inp.size(0))

            for i, logits in enumerate(prev_layers_output):
                acc1 = accuracy(logits, target, topk=(1,))[0]
                prev_layers_top1[i].update(acc1, inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if args.rank == 0:
        logger.info(
            f"Test:\t Time {batch_time.avg}\t"
            f"Loss {losses.avg}\t"
            f"Acc@1 {top1.avg}\t"
            f"Acc@5 {top5.avg}")

    return losses.avg, top1.avg, top5.avg, *[m.avg for m in prev_layers_top1], ensemble_top1.avg


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Suave', parents=[get_args_parser()])
    global args
    args = parser.parse_args()
    train_suave(args)
