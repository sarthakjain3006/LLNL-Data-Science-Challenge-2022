################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################

import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from hd5_explore import hdf_read
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, Subset

from model import Model_3DCNN, strip_prefix_if_present
#from model_simple import CNNModel
#from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, mean_squared_error, mean_absolute_error, r2_score

from file_util import *
import math
from scipy import stats
import torch.nn.functional as F

# set CUDA for PyTorch
use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()
device_name = 'cuda:0'
print(cuda_count)
if use_cuda:
	device = torch.device(device_name)
	torch.cuda.set_device(int(device_name.split(':')[1]))
else:
	device = torch.device("cpu")
print(use_cuda, cuda_count, device)


def worker_init_fn(worker_id):
	np.random.seed(int(0))

def print_complete():
	print('Done!')

def train():

	# Read hd5 files for training and validation
	X_train,Y_train = hdf_read('postera_protease2_pos_neg_train.hdf5')
	train_data = TensorDataset(X_train,Y_train)

	X_val,Y_val = hdf_read('postera_protease2_pos_neg_val.hdf5')
	val_data = TensorDataset(X_val,Y_val)

	# check multi-gpus
	
	num_workers = 0
	if cuda_count > 1:
		num_workers = cuda_count

	batch_size = 128
	epoch_count = 1
	checkpoint_iter = 10
	# initialize data loader
	batch_count = len(Y_train) // batch_size
	dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None)


	val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

	# define voxelizer, gaussian_filter
	voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose = 0)
	gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)
	# define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose = 1)
	if use_cuda:
		model = model.cuda()
	if cuda_count > 1:
		model = nn.DataParallel(model)

	model=model.to(device)
	
	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	best_metric_value = 0.0
	best_model = None
	input_metric = {
        'y_true': None,
        'y_pred': None,
        }

	loss_fn = nn.BCEWithLogitsLoss()
	metric = precision_score
	optimizer = Adam(model.parameters(), lr= 0.00001)
	#optimizer = RMSprop(model.parameters(), lr = 0.0001)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

	# load model
	epoch_start = 0
	model_path = 'refined_model.pth'
	if valid_file(model_path):
		checkpoint = torch.load(model_path, map_location=device)
		model_state_dict = checkpoint.pop("model_state_dict")
		strip_prefix_if_present(model_state_dict, "module.")
		model_to_save.load_state_dict(model_state_dict, strict=False)
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		epoch_start = checkpoint["epoch"]
		loss = checkpoint["loss"]
		print("checkpoint loaded: %s" % model_path)


	step = 0
	for epoch_ind in range(epoch_start,epoch_count):
		vol_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
		losses = []
		accuracy = []
		model.train()
		for batch_ind,batch in enumerate(dataloader):

		
			x_batch_cpu, y_batch_cpu = batch[0], batch[1]
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
			
			# voxelize into 3d volume
			for i in range(x_batch.shape[0]):
				xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
				vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
			vol_batch = gaussian_filter(vol_batch)
			
			# forward training
			ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])
			input_metric['y_true'] = y_batch.cpu().detach().numpy()
			input_metric['y_pred'] = np.digitize(ypred_batch.cpu().detach().numpy(),[0.5])

			training_metric = metric(**input_metric)


			# compute loss
			
			loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.cpu().float())
	
			losses.append(loss.cpu().data.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			print("[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f" % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item(), optimizer.param_groups[0]['lr']))
			
			if step % checkpoint_iter == 0:
				correct = 0
				total = 0
				checkpoint_dict = {
					"model_state_dict": model_to_save.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": loss,
					"step": step,
					"epoch": epoch_ind
				}
				torch.save(checkpoint_dict, model_path)
				print("checkpoint saved: %s" % model_path)
			step += 1
		print(f"Training - Epoch {epoch_ind + 1}/{epoch_count}, Batch: {batch_ind + 1}/{epoch_count}, Loss: {loss:.3f} Metric:{training_metric:.3f}")
		print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, epoch_count, np.mean(losses)))

	

		val_losses = []
		val_accuracy = []
		model.eval()
		with torch.no_grad():
			for batch_ind, batch in enumerate(val_dataloader):

				x_batch_cpu, y_batch_cpu = batch
				x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
					
				for i in range(x_batch.shape[0]):
					xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
					vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
				vol_batch = gaussian_filter(vol_batch)
					
				ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])


				loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.cpu().float())
						
				val_losses.append(loss.cpu().data.item())
				print("[%d/%d-%d/%d] validation, loss: %.3f" % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item()))

			print("[%d/%d] validation, epoch loss: %.3f" % (epoch_ind+1, epoch_count, np.mean(val_losses)))

	# close dataset

def test():


	X_test,Y_test = hdf_read('postera_protease2_pos_neg_test.hdf5')
	test_data = TensorDataset(X_test,Y_test)

	# check multi-gpus
	num_workers = 0
	if cuda_count > 1:
		num_workers = cuda_count

	# initialize data loader
	batch_size = 32
	batch_count = len(Y_test) // batch_size
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

	# define voxelizer, gaussian_filter
	voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=0)
	gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=5, sigma=1, use_cuda=use_cuda)

	# define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose=0)
	#if use_cuda:
	#	model = model.cuda()
	if cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)

	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	# load model
	model_path = 'refined_model.pth'
	checkpoint = torch.load(model_path, map_location=device)
	#checkpoint = torch.load(args.model_path)
	model_state_dict = checkpoint.pop("model_state_dict")
	strip_prefix_if_present(model_state_dict, "module.")
	model_to_save.load_state_dict(model_state_dict, strict=False)
	output_dir = os.path.dirname(model_path)

	vol_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
	ytrue_arr = np.zeros((len(Y_test),), dtype=np.float32)
	ypred_arr = np.zeros((len(Y_test),), dtype=np.float32)
	zfeat_arr = np.zeros((len(Y_test), 100), dtype=np.float32)
	pred_list = []

	model.eval()
	with torch.no_grad():
		for bind, batch in enumerate(test_dataloader):
		
			# transfer to GPU
			x_batch_cpu, y_batch_cpu = batch
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
			
			# voxelize into 3d volume
			bsize = x_batch.shape[0]
			for i in range(bsize):
				xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
				vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
			vol_batch = gaussian_filter(vol_batch)
			
			# forward training
			ypred_batch, zfeat_batch = model(vol_batch[:x_batch.shape[0]])

			ytrue = y_batch_cpu.float().data.numpy()[:,0]
			ypred = ypred_batch.cpu().float().data.numpy()[:,0]
			zfeat = zfeat_batch.cpu().float().data.numpy()
			ytrue_arr[bind*batch_size:bind*batch_size+bsize] = ytrue
			ypred_arr[bind*batch_size:bind*batch_size+bsize] = ypred
			zfeat_arr[bind*batch_size:bind*batch_size+bsize] = zfeat


			print("[%d/%d] evaluating" % (bind+1, batch_count))
	
		
	rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
	mae = mean_absolute_error(ytrue_arr, ypred_arr)
	r2 = r2_score(ytrue_arr, ypred_arr)
	pearson, ppval = stats.pearsonr(ytrue_arr, ypred_arr)
	spearman, spval = stats.spearmanr(ytrue_arr, ypred_arr)
	mean = np.mean(ypred_arr)
	std = np.std(ypred_arr)
	print("Evaluation Summary:")
	print("RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % (rmse, mae, r2, pearson, spearman, mean, std))






def main():
	train()
#	plt.figure(figsize=(10,5))
#	plt.title("Training and Validation Loss")
#	plt.plot(val_losses,label="val")
#	plt.plot(train_losses,label="train")
#	plt.xlabel("iterations")
#	plt.ylabel("Loss")
#	plt.legend()
#	plt.show()
	print_complete()
	test()
	print_complete()

if __name__ == "__main__":
	main()

	
