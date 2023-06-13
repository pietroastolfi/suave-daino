# Based on https://github.com/facebookresearch/dino/blob/main/main_dino.py

import pathlib
import sys
_parentdir = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_parentdir))

import argparse
import os
import datetime
import time
import math
import json
from pathlib import Path
import wandb

import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.cuda.amp
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import src.vision_transformer as vits
from src.vision_transformer import DINOHead
from src.mixup import MultiViewCollateMixup, CollateMixup, collate_rep_aug
from src import augment

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('Daino', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=1000, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # teacher temperature
    parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
    
    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument("--lab_batch_size_per_gpu", default=50, type=int,
        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=0, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument("--init_proj", type=utils.bool_flag, default=False,
        help="whether to initialize the dino head from the pretrained model")
    parser.add_argument("--dino_init_ckpt", type=str, default="dino_deitsmall16_pretrain_full_checkpoint.pth",
        help="Checkpoint used to iniziatile DINO")
    parser.add_argument("--label_smoothing", type=float, default=0.01,
                    help="label smoothing for supervised loss")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Data
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
                        help="fine-tune on either 1% or 10% of labels")
    parser.add_argument("--mixup_unlab", type=utils.bool_flag, default=False,
                        help="whether to use mixup on unlabeled images")
    parser.add_argument("--mixup_unlab_type", type=str, default="all", choices=["all", "globals"],
                        help="whether to use mixup on all views or on global views only")
    parser.add_argument("--mixup_lab", type=utils.bool_flag, default=False,
                        help="whether to use mixup on labeled images")
    parser.add_argument("--mixup_lab_type", type=str, default="extend", choices=["extend", "replace"],
                        help="whether to extend or replace the batch with mixed imgs")
    parser.add_argument("--mixup_lab_prob", type=float, default=0.5,
                        help="probability of applying mixup when mixup replace is used")

    
    # Misc
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--wandb", type=utils.bool_flag, default=False, help="whether to use wandb or not")
    return parser


def train_daino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # init wandb
    if args.wandb and args.rank == 0:
        wandb_id = os.path.split(args.output_dir)[-1]
        print(f"WANDB active ID: {wandb_id}")
        name = wandb_id.split("-", 1)[-1]
        wandb.init(
            project="simple-semi-self",
            id=wandb_id,
            config=args,
            resume="allow",
            name=name,
            save_code=True,
        )

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    train_data_path = os.path.join(args.data_path, "train")
    val_data_path = os.path.join(args.data_path, "val")

    dataset = ImageFolderInverted(train_data_path, transform=transform)
    # dataset.samples = dataset.samples[:1000]

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
            num_classes=args.out_dim,
            views_to_mix=args.mixup_unlab_type
        )
    else:
        collate_fn = None
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    print(f"Data loaded: there are {len(dataset)} images.")

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
            num_classes=args.out_dim,
        )
    else:
        collate_fn = collate_rep_aug
    lab_sampler = torch.utils.data.distributed.DistributedSampler(lab_dataset)
    lab_loader = torch.utils.data.DataLoader(
        lab_dataset,
        sampler=lab_sampler,
        batch_size=args.lab_batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    print(f"Labeled data loaded: there are {len(lab_dataset)} images.")
    # lab_iterable = (x for _ in count() for x in lab_loader)
    lab_iterable = utils.cycle(lab_loader)

    # ============ build validation data ... ============
    val_dataset = datasets.ImageFolder(val_data_path)
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # start from pretrained dino
    dino_ckpt = torch.load(args.dino_init_ckpt, map_location="cpu")
    del dino_ckpt["student"]["module.head.last_layer.weight_g"]
    del dino_ckpt["student"]["module.head.last_layer.weight_v"]   # skip prototypes
    if not args.init_proj:  # skip projection head
        proj_names = [name for name in dino_ckpt["student"].keys() if "module.head" in name]
        for name in proj_names:
            del dino_ckpt["student"][name]
    ckpt_loaded = student.load_state_dict(dino_ckpt["student"], strict=False)
    print(ckpt_loaded)
    del dino_ckpt   # free up memory

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.temperature,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    validate(val_loader, student, args)  # debug validation

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, lab_iterable, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)
    
        val_stats = validate(val_loader, teacher, args)

        # ============ writing logs ... ============
        if args.wandb and args.rank == 0:
            wandb.log({**train_stats, **val_stats})


        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    print("teacher", validate(val_loader, teacher, args))
    print("student", validate(val_loader, student, args))

    if args.wandb and args.rank == 0:
        wandb.finish()

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, unlab_loader, lab_iterable,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (unlab_labels, *images) in enumerate(metric_logger.log_every(unlab_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(unlab_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        mixup_unlab_lams = None
        if args.mixup_unlab:
            mixup_unlab_lams = images[1]
            mixup_unlab_idxs = images[2]

        lab_images, lab_labels = next(lab_iterable)
        lab_images = lab_images.cuda(non_blocking=True)
        lab_labels = lab_labels.cuda(non_blocking=True)

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images[0]]
        unlab_labels = unlab_labels.cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student([lab_images] + images)

            # split student output
            lab_bs, bs = lab_images.size(0), images[0].size(0)
            lab_student_output, unlab_student_output = student_output[:lab_bs], student_output[lab_bs:]
            unlab_loss = dino_loss(unlab_student_output, teacher_output, epoch)


        unlab_acc1 = 0
        for q in dino_loss.pseudo_unlab_labels:
            tmp_acc1, tmp_acc5 = utils.accuracy(q, unlab_labels, topk=(1, 5))
            unlab_acc1 += tmp_acc1    

        unlab_acc1 /= len(dino_loss.pseudo_unlab_labels)

        # TODO: Mixup unlab
        mixed_unlab_bs = 0
        mixed_unlab_loss = 0

        # ============ supervised loss ... ============
        lab_loss = F.cross_entropy(lab_student_output / args.temperature, lab_labels, label_smoothing=args.label_smoothing)
        # lab_loss = 0
        lab_acc1, lab_acc5 = utils.accuracy(lab_student_output, lab_labels, topk=(1, 5))

        total_bs = (bs * (args.local_crops_number + 2) + lab_bs + mixed_unlab_bs) 
        lab_weight = lab_bs / total_bs
        unlab_weight = bs * (args.local_crops_number + 2) / total_bs
        mixed_unlab_weight = mixed_unlab_bs / total_bs
        loss = (
            unlab_weight * unlab_loss + \
            lab_weight * lab_loss + \
            mixed_unlab_weight * mixed_unlab_loss
        )

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(unlab_loss=unlab_loss.item())
        metric_logger.update(lab_loss=lab_loss.item())
        metric_logger.update(unlab_top1=unlab_acc1)
        metric_logger.update(lab_top1=lab_acc1)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(val_loader, model, args):

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        for inp, target in val_loader:

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(inp)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            metric_logger.update(val_loss=loss.item())
            metric_logger.update(val_top1=acc1)

    if args.rank == 0:
        print("Test\nAveraged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        self.pseudo_unlab_labels = teacher_out

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            augment.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            augment.GaussianBlur(0.1),
            augment.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            augment.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class ImageFolderInverted(datasets.ImageFolder):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        label = sample[1]
        img = sample[0]
        return label, img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Daino', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_daino(args)
