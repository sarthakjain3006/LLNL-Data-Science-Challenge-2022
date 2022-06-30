import numpy as np
import pandas as pd
import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

from model import Model_3DCNN, strip_prefix_if_present
from img_util import GaussianFilter, Voxelizer3D
from file_util import *


def train():
    # load dataset
    if args.complex_type == 1:
        is_crystal = True
    else:
        is_crystal = False
    dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type,
                            os.path.join(args.data_dir, args.csv_fn), is_crystal=is_crystal,
                            rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold)

    # if validation set is available
    val_dataset = None
    if len(args.vmlhdf_fn) > 0:
        val_dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.vmlhdf_fn), args.dataset_type,
                                    os.path.join(args.data_dir, args.vcsv_fn), is_crystal=is_crystal,
                                    rmsd_weight=args.rmsd_weight, rmsd_thres=args.rmsd_threshold)

    # check multi-gpus
    num_workers = 0
    if args.multi_gpus and cuda_count > 1:
        num_workers = cuda_count

    # initialize data loader
    batch_count = len(dataset) // args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                            worker_init_fn=None)

    # if validation set is available
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                                    worker_init_fn=None)

    # define voxelizer, gaussian_filter
    voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=args.verbose)
    gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)

    # define model
    model = Model_3DCNN(use_cuda=use_cuda, verbose=args.verbose)
    # if use_cuda:
    #	model = model.cuda()
    if args.multi_gpus and cuda_count > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model_to_save = model.module
    else:
        model_to_save = model

    # set loss, optimizer, decay, other parameters
    if args.rmsd_weight == True:
        loss_fn = WeightedMSELoss().float()
    else:
        loss_fn = nn.MSELoss().float()
    # optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    optimizer = RMSprop(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)

    # load model
    epoch_start = 0
    if valid_file(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        # checkpoint = torch.load(args.model_path)
        model_state_dict = checkpoint.pop("model_state_dict")
        strip_prefix_if_present(model_state_dict, "module.")
        model_to_save.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print("checkpoint loaded: %s" % args.model_path)

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    output_dir = os.path.dirname(args.model_path)

    step = 0
    for epoch_ind in range(epoch_start, args.epoch_count):
        vol_batch = torch.zeros((args.batch_size, 19, 48, 48, 48)).float().to(device)
        losses = []
        model.train()
        for batch_ind, batch in enumerate(dataloader):

            # transfer to GPU
            if args.rmsd_weight == True:
                x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
            else:
                x_batch_cpu, y_batch_cpu = batch
            x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)

            # voxelize into 3d volume
            for i in range(x_batch.shape[0]):
                xyz, feat = x_batch[i, :, :3], x_batch[i, :, 3:]
                vol_batch[i, :, :, :, :] = voxelizer(xyz, feat)
            vol_batch = gaussian_filter(vol_batch)

            # forward training
            ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])

            # compute loss
            if args.rmsd_weight == True:
                loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float(), w_batch_cpu.float())
            else:
                loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())

            losses.append(loss.cpu().data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print("[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f" % (
            epoch_ind + 1, args.epoch_count, batch_ind + 1, batch_count, loss.cpu().data.item(),
            optimizer.param_groups[0]['lr']))
            if step % args.checkpoint_iter == 0:
                checkpoint_dict = {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "step": step,
                    "epoch": epoch_ind
                }
                torch.save(checkpoint_dict, args.model_path)
                print("checkpoint saved: %s" % args.model_path)
            step += 1

        print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind + 1, args.epoch_count, np.mean(losses)))

        if val_dataset:
            val_losses = []
            model.eval()
            with torch.no_grad():
                for batch_ind, batch in enumerate(val_dataloader):
                    if args.rmsd_weight == True:
                        x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
                    else:
                        x_batch_cpu, y_batch_cpu = batch
                    x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)

                    for i in range(x_batch.shape[0]):
                        xyz, feat = x_batch[i, :, :3], x_batch[i, :, 3:]
                        vol_batch[i, :, :, :, :] = voxelizer(xyz, feat)
                    vol_batch = gaussian_filter(vol_batch)

                    ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])

                    if args.rmsd_weight == True:
                        loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float(), w_batch_cpu.float())
                    else:
                        loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())

                    val_losses.append(loss.cpu().data.item())
                    print("[%d/%d-%d/%d] validation, loss: %.3f" % (
                    epoch_ind + 1, args.epoch_count, batch_ind + 1, batch_count, loss.cpu().data.item()))

                print("[%d/%d] validation, epoch loss: %.3f" % (epoch_ind + 1, args.epoch_count, np.mean(val_losses)))

    # close dataset
    dataset.close()
    val_dataset.close()