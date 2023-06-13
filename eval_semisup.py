# Based on https://github.com/facebookresearch/swav/blob/main/eval_semisup.py

import argparse
import math
import os
import time
from logging import getLogger
import urllib
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
)
import src.resnet50 as resnet_models
from src.mixup import CollateMixup

logger = getLogger()


parser = argparse.ArgumentParser(description="Evaluate models: Fine-tuning with 1% or 10% labels on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
                    help="fine-tune on either 1% or 10% of labels")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to imagenet")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--wandb", type=bool_flag, default=False, help="whether to use wandb or not")


#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--hidden_mlp", default=[2048], type=int, nargs="+",
                    help="hidden layer dimension in projection head")
parser.add_argument("--load_hidden_mlp", type=bool_flag, default=True,
                    help="whether to load the weights for the proj head layers that match shapes")
parser.add_argument("--ensemble_eval", type=bool_flag, default=False,
                    help="whether to train a linear classifier on top of features ensemble(concat)")
parser.add_argument("--bn_ensemble", type=bool_flag, default=False,
                    help="whether to use batch norm prior to the ensemble linear layer")


#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=50, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.02, type=float, help="initial learning rate - trunk")
parser.add_argument("--lr_last_layer", default=0.02, type=float, help="initial learning rate - head")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate value")
parser.add_argument("--warmup_epochs", default=0, type=int, help="number of warmup epochs")
parser.add_argument("--wd", default=0., type=float, help="weight decay")

#########################
#### dist parameters ####
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
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
parser.add_argument("--mixup", type=bool_flag, default=False,
                    help="whether to use mixup on labeled images")
parser.add_argument("--mixup_type", type=str, default="extend", choices=["extend", "replace"],
                    help="whether to extend or replace the batch with mixed imgs")
parser.add_argument("--light_augment", type=bool_flag, default=False,
                    help="""whether to use additional data augmentation with low intensity, 
                        i.e., color jit (0.4) with p = 0.2, and greyscale with p = 0.2""")

def main():
    global args, best_acc
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args,
        "epoch",
        "train_loss",
        "train_acc1",
        "train_acc5",
        "val_loss",
        "val_acc1",
        "val_acc5",
    )

    # init wandb
    if args.wandb and args.rank == 0:
        wandb_id = os.path.split(args.dump_path)[-1]
        logger.info(f"WANDB active ID: {wandb_id}")
        name = "FT-" + wandb_id.split("-", 1)[-1]
        wandb.init(
            project="semi-self",
            entity="unitn-mhug",
            id=wandb_id,
            config=args,
            resume="allow",
            name=name,
            save_code=True,
        )

    # build data
    train_data_path = os.path.join(args.data_path, "train")
    train_dataset = datasets.ImageFolder(train_data_path)
    # take either 1% or 10% of images
    subset_file = urllib.request.urlopen("https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/" + str(args.labels_perc) + "percent.txt")
    list_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]
    list_img_classes = []
    for i in range(len(list_imgs)):
        list_img_classes.append([
            k for k in train_dataset.class_to_idx.keys()
            if k.startswith(list_imgs[i].split('_')[0])
        ][0])
    train_dataset.samples = [(
        os.path.join(train_data_path, lic, li),
        train_dataset.class_to_idx[lic]
    ) for lic, li in zip(list_img_classes, list_imgs)]
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    if args.light_augment:
        lt_aug = [
            transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=0.8),
        ]
    else:
        lt_aug = []
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        *lt_aug,
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])

    if args.mixup:
        collate_fn = CollateMixup(
            extend_batch=True if args.mixup_type == "extend" else False,
            mixup_alpha=.8,
            cutmix_alpha=1.,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            correct_lam=True,
            label_smoothing=0.0,
            num_classes=len(train_dataset.classes),
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
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        hidden_mlp=args.hidden_mlp,
        output_dim=1000,
        ensemble_eval=args.ensemble_eval,
        bn_ensemble=args.bn_ensemble,
    )

    # convert batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # TODO renaming for old ckpt compatibility
        # change the name of the last layer of the projection
        for i in range(1, 10):
            state_dict = {
                k.replace(f"projection_head.{i}.weight", f"projection_head.{i}.0.weight"): v 
                for k, v in state_dict.items()
            }
            state_dict = {
                k.replace(f"projection_head.{i}.bias", f"projection_head.{i}.0.bias"): v 
                for k, v in state_dict.items()
            } 

        for k, v in model.state_dict().items():

            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif not args.load_hidden_mlp and "projection_head" in k:
                logger.info('key "{}" intentionally not loaded'.format(k))
                state_dict[k] = v
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training from random weights")

    # model to gpu
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # set optimizer
    trunk_parameters = []
    head_parameters = []
    for name, param in model.named_parameters():
        if 'head' in name:
            head_parameters.append(param)
        else:
            trunk_parameters.append(param)
    optimizer = torch.optim.SGD(
        [{'params': trunk_parameters},
         {'params': head_parameters, 'lr': args.lr_last_layer}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    # set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, args.decay_epochs, gamma=args.gamma
    # )
    warmup_lr_schedule = np.linspace(0, args.lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": (0., 0.)}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        # scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        train_loader.sampler.set_epoch(epoch)

        scores = train(model, optimizer, train_loader, epoch, lr_schedule)
        scores_val = validate_network(val_loader, model)
        training_stats.update(scores + scores_val)

        # save checkpoint
        if args.rank == 0:
            if args.wandb:
                # get last row of pd dataframe and trasform into a dict {col1: v1, col2: v2}
                wandb.log(training_stats.stats.iloc[-1:].to_dict("records")[-1])

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.dump_path, "checkpoint.pth.tar"))
    logger.info("Fine-tuning with {}% of labels completed.\n"
                "Test accuracies: top-1 {acc1:.1f}, top-5 {acc5:.1f}".format(
                args.labels_perc, acc1=best_acc[0], acc5=best_acc[1]))
    
    if args.wandb and args.rank == 0:
        wandb.finish()


def train(model, optimizer, loader, epoch, lr_schedule):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # update learning rate
        iteration = epoch * len(loader) + iter_epoch
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # forward
        output, _ = model(inp)

        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1, inp.size(0))
        top5.update(acc5, inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR trunk {lr}\t"
                "LR head {lr_W}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                    lr_W=optimizer.param_groups[1]["lr"],
                )
            )
    return epoch, losses.avg, top1.avg, top5.avg


def validate_network(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc

    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, _ = model(inp)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1, inp.size(0))
            top5.update(acc5, inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg > best_acc[0]:
        best_acc = (top1.avg, top5.avg)

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc[0]))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
