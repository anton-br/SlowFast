#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    regr_list = []
    num_list = []
    top_list = []
    for cur_iter, (inputs, labels, _) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        if isinstance(labels, (list,)):
            for i in range(len(labels)):
                labels[i] = labels[i].cuda(non_blocking=True)
            labels = torch.stack((labels))
        else:
            labels = labels.cuda(non_blocking=True)

        if cfg.MODEL.LOSS_FUNC == 'mse':
            labels = labels.float()

        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)


        # Perform the forward pass.
        preds = model(inputs)
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        # Compute the loss.

        loss = loss_fun(preds, labels)
        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        top1_err = None
        # Compute the errors.
        num_classes = cfg.MODEL.NUM_CLASSES
        if cfg.DATA.LABELS_TYPE == 'regression':
            ln = (labels.size(1) - 1) // 2 + 1
            pr = preds[:,ln:].reshape(-1, 5)
            lb = labels[:,ln:].reshape(-1)
            num_topks_correct = metrics.topks_correct(pr, lb, (1, ))
            top1_err = (1.0 - num_topks_correct[0] / len(lb)) * 100.0
            regr = ((preds[:, 0] - labels[:, 0])**2).mean()
            numbers =  ((preds[:, 1: ln] - labels[:, 1: ln])**2).mean()
            if cfg.NUM_GPUS > 1:
                regr, numbers = du.all_reduce([regr, numbers])
            regr_list.append(regr.item())
            num_list.append(numbers.item())
        elif cfg.DATA.LABELS_TYPE == 'length':
            regr = ((preds[:, 0] - labels[:, 0])**2).mean()
            numbers = ((preds[:, 1: ] - labels[:, 1: ])**2).mean()
            if cfg.NUM_GPUS > 1:
                regr, numbers = du.all_reduce([regr, numbers])
            regr_list.append(regr.item())
            num_list.append(numbers.item())
            num_topks_correct = metrics.topks_correct(preds, labels, (1, ))
            top1_err = num_topks_correct[0] * 0.0
        elif cfg.DATA.LABELS_TYPE == 'stend':
            top1_err = loss.clone()
            # sigm = torch.nn.Sigmoid()
            # start = sigm(preds[:, 0]).cpu().detach().numpy()
            # end = sigm(preds[:, 1]).cpu().detach().numpy()

        else:
            num_topks_correct = metrics.topks_correct(preds, labels, (1, ))
            preds_ix = preds.size(2)*preds.size(0) if cfg.DATA.LABELS_TYPE == 'mask' else preds.size(1)
            top1_err = (1.0 - num_topks_correct[0] / preds_size) * 100.0

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err  = du.all_reduce(
                [loss, top1_err]
            )

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err = (
            loss.item(),
            top1_err.item()
        )
        top_list.append(top1_err)
        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            top1_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
        )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    if cfg.DATA.LABELS_TYPE == 'regression' or cfg.DATA.LABELS_TYPE == 'length':
        print('---------------------')
        print(f'LOSS VALUES!!: SIZE_LOSS:{np.mean(regr_list)} NUM_LOSS:{np.mean(num_list)} CLASS_LOSS:{np.mean(top_list)}')
        print('---------------------')
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    regr_list = []
    num_list = []
    top_list = []
    for cur_iter, (inputs, labels, _) in enumerate(val_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        if isinstance(labels, (list,)):
            for i in range(len(labels)):
                labels[i] = labels[i].cuda(non_blocking=True)
            labels = torch.stack((labels))
        else:
            labels = labels.cuda(non_blocking=True)


        preds = model(inputs)

        if cfg.DATA.LABELS_TYPE == 'regression':
            pr = preds[:,ln:].reshape(-1, 5)
            lb = labels[:,ln:].reshape(-1)
            num_topks_correct = metrics.topks_correct(pr, lb, (1, ))
            top1_err = (1.0 - num_topks_correct[0] / len(lb)) * 100.0
            regr = ((preds[:, 0] - labels[:, 0])**2).mean()
            numbers =  ((preds[:, 1: ln] - labels[:, 1: ln])**2).mean()
            if cfg.NUM_GPUS > 1:
                regr, numbers = du.all_reduce([regr, numbers])
            regr_list.append(regr.item())
            num_list.append(numbers.item())
        elif cfg.DATA.LABELS_TYPE == 'length':
            regr = ((preds[:, 0] - labels[:, 0])**2).mean()
            numbers =  ((preds[:, 1: ] - labels[:, 1: ])**2).mean()
            if cfg.NUM_GPUS > 1:
                regr, numbers = du.all_reduce([regr, numbers])
            regr_list.append(regr.item())
            num_list.append(numbers.item())
            num_topks_correct = metrics.topks_correct(preds, labels, (1, ))
            top1_err = num_topks_correct[0] * 0.0
        elif cfg.DATA.LABELS_TYPE == 'stend':
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            loss = loss_fun(preds, labels)
            top1_err = loss.clone()
        else:
            num_topks_correct = metrics.topks_correct(preds, labels, (1, ))
            preds_ix = preds.size(2)*preds.size(0) if cfg.DATA.LABELS_TYPE == 'mask' else preds.size(1)
            top1_err = (1.0 - num_topks_correct[0] / preds_size) * 100.0

        # num_topks_correct = metrics.topks_correct(preds, labels, (1, ))
        # # Combine the errors across the GPUs.
        # preds_ix = 2 if cfg.DATA.LABELS_TYPE == 'mask' else 1
        # top1_err = (1.0 - num_topks_correct[0] / preds.size(preds_ix)) * 100.0

        if cfg.NUM_GPUS > 1:
            top1_err = du.all_reduce([top1_err])[0]
        # Copy the errors from GPU to CPU (sync point).
        top1_err = top1_err.item()
        top_list.append(top1_err)
        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            top1_err, inputs[0].size(0) * cfg.NUM_GPUS
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()
    if cfg.DATA.LABELS_TYPE == 'regression' or cfg.DATA.LABELS_TYPE == 'length':
        print('---------------------')
        print(f'VALIDATE LOSS!!: SIZE_LOSS:{np.mean(regr_list):.5} NUM_LOSS:{np.mean(num_list):.5} CLASS_LOSS:{np.mean(top_list):.5}')
        print('---------------------')
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)


    # Print config.
    logger.info("Train with config:")
    # logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, is_train=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(model)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)

    return model