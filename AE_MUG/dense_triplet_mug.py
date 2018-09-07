from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
import time
from glob import glob
#from util import *
import numpy as np
from PIL import Image

import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import gradcheck
from torch.autograd import Function
import math
# our data loader
import MUGloader
import gc
import WaspNet_dany as WaspNet

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=25, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default = True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--modelPath', default='', help="path to model (to continue training)")
parser.add_argument('--dirCheckpoints', default='.', help='folder to model checkpoints')
parser.add_argument('--dirImageoutput', default='.', help='folder to output images')
parser.add_argument('--dirTestingoutput', default='.', help='folder to testing results/images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epoch_iter', type=int,default=1000, help='number of epochs on entire dataset')
parser.add_argument('--location', type = int, default=0, help ='where is the code running')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
parser.add_argument('--nc', type=int, default=3, help='num channels')
parser.add_argument('--zdim', type=int, default=12, help='latent variable size')
parser.add_argument('--edim', type=int, default=6, help='dimensions of expression vec')
parser.add_argument('--pdim', type=int, default=6, help='dimensions of person vec')

opt = parser.parse_args()
print(opt)

## do not change the data directory
opt.data_dir_prefix = '/nfs/bigdisk/zhshu/data/fare/'

## change the output directory to your own
opt.output_dir_prefix = '/nfs/bigdisk/peterli/AE_triplet_results'
opt.dirCheckpoints	= opt.output_dir_prefix + '/checkpoints'
opt.dirImageoutput	= opt.output_dir_prefix + '/images'
opt.dirTestingoutput  = opt.output_dir_prefix + '/testing'

opt.imgSize = 64

try:
	os.makedirs(opt.dirCheckpoints)
except OSError:
	pass
try:
	os.makedirs(opt.dirImageoutput)
except OSError:
	pass
try:
	os.makedirs(opt.dirTestingoutput)
except OSError:
	pass


# sample iamges
def visualizeAsImages(img_list, output_dir,
					  n_sample=4, id_sample=None, dim=-1,
					  filename='myimage', nrow=2,
					  normalize=False):
	if id_sample is None:
		images = img_list[0:n_sample,:,:,:]
	else:
		images = img_list[id_sample,:,:,:]
	if dim >= 0:
		images = images[:,dim,:,:].unsqueeze(1)
	vutils.save_image(images,
		'%s/%s'% (output_dir, filename+'.png'),
		nrow=nrow, normalize = normalize, padding=2)


def parseSampledDataTriplet(dp0_img,  dp9_img, dp1_img):
	###
	dp0_img  = dp0_img.float()/255 # convert to float and rerange to [0,1]
	dp0_img  = dp0_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
	###
	dp9_img  = dp9_img.float()/255 # convert to float and rerange to [0,1]
	dp9_img  = dp9_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
	###
	dp1_img  = dp1_img.float()/255 # convert to float and rerange to [0,1]
	dp1_img  = dp1_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
	return dp0_img, dp9_img, dp1_img


def setFloat(*args):
	barg = []
	for arg in args:
		barg.append(arg.float())
	return barg

def setCuda(*args):
	barg = []
	for arg in args:
		barg.append(arg.cuda())
	return barg

def setAsVariable(*args):
	barg = []
	for arg in args:
		barg.append(Variable(arg))
	return barg

def setAsDumbVariable(*args):
	barg = []
	for arg in args:
		barg.append(Variable(arg,requires_grad=False))
	return barg



# Multipie training data folder list
MultipieData = []
#session 01

MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_01_select/')
MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_02_select/')
MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_03_select/')
MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_04_select/')
MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_05_select/')
MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_06_select/')
MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_07_select/')


# #session 02
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_01_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_02_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_03_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_04_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_05_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_06_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_07_select/')
# #session 03
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_01_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_02_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_03_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_04_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_05_select/')
# #session 04
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_01_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_02_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_03_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_04_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_05_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_06_select/')
# MultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_07_select/')

# Small Testing Set
# TestingMultipieData = []
# TestingMultipieData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_select_test/')

# MUG data

MugData = []

MugData_root = '/home/peterli/faces/AE_MUG/MUG_data'

# for root, dirlist, filelist in os.walk(MugData_root):
# 	for person_dir in dirlist:
# 		MugData.append(person_dir)


class AE(nn.Module):
	def __init__(self, latent_variable_size):
		super(AE, self).__init__()
		#self.latent_variable_size = latent_variable_size

		self.encoder = WaspNet.Dense_Encoders_AE_SliceSplit(opt)
		self.decoder = WaspNet.Dense_Decoders_AE(opt)

	def forward(self, x):
		z, z_per, z_exp = self.encoder(x)
		recon_x = self.decoder(z)
		return recon_x, z, z_per, z_exp

model=AE(latent_variable_size=128)


if opt.cuda:
	 model.cuda()

def recon_loss_func(recon_x, x):
	recon_func = nn.MSELoss()
	return recon_func(recon_x, x)

def cosine_loss_func(z1, z2, label):
	cosine_func = nn.CosineEmbeddingLoss()
	cosine_func.margin = 0.5
	#y = torch.ones_like(z2)
	y = torch.ones(z1.size()[0]).cuda()

	#size of target has to match size of inputs
	y.requires_grad_(False)
	if label == 1: # measure similarity
		return cosine_func(z1, z2, target=y)
	elif label == -1: # measure dissimilarity
		y = y * -1
		return cosine_func(z1, z2, target=y)

def triplet_loss_func(a, p, n):
	triplet_func = nn.TripletMarginLoss()
	return triplet_func(a, p, n)

def L1(x, y):
	return torch.mean(torch.abs(x - y))

def BCE(x, target):
	BCE_func = nn.BCEWithLogitsLoss() # combined sigmoid and BCE into one layer
	return BCE_func(x, target)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

lossfile = open(opt.output_dir_prefix + "/losses.txt", "w")
sim_loss = 0
dis_loss = 0


def train(epoch):
	print("train")
	model.train()
	recon_pie_train_loss = 0
	recon_mug_train_loss = 0
	cosine_train_loss = 0
	triplet_train_loss = 0
	swap_train_loss = 0
	expression_train_loss = 0
	inten_train_loss = 0
	pie_dataroot = random.sample(MultipieData,1)[0]
	# mug_dataroot = random.sample(MugData,1)[0]  # why is it without replacement?

	dataset = MUGloader.TrainTestSplit(opt, pieroot=pie_dataroot, mugroot=MugData_root, resize=64)

	print('# size of the current (sub)dataset is %d' %len(dataset))
 #   train_amount = train_amount + len(dataset)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

	for batch_idx, data_point in enumerate(dataloader, 0):

		gc.collect() # collect garbage

		###### PIE ######
		# sample the data points:
		# dp0_img: image of data point 0
		# dp9_img: image of data point 9, which is different in ``expression'' compare to dp0, same person as dp0
		# dp1_img: image of data point 1, which is different in ``person'' compare to dp0, same expression as dp0
		dp0_img, dp9_img, dp1_img, dp0_ide, dp9_ide, dp1_ide, dp2_img, dp3_img, dp4_img, inten2, inten3, inten4 = data_point
		dp0_img, dp9_img, dp1_img = parseSampledDataTriplet(dp0_img, dp9_img, dp1_img)
		dp2_img, dp3_img, dp4_img = parseSampledDataTriplet(dp2_img, dp3_img, dp4_img)

		if opt.cuda:
			dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
			dp2_img, dp3_img, dp4_img = setCuda(dp2_img, dp3_img, dp4_img)
		dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img)
		dp2_img, dp3_img, dp4_img = setAsVariable(dp2_img, dp3_img, dp4_img)

		recon_batch_dp9, z_dp9, z_per_dp9, z_exp_dp9 = model(dp9_img)
		recon_batch_dp1, z_dp1, z_per_dp1, z_exp_dp1 = model(dp1_img)

		recon_batch_dp0, z_dp0, z_per_dp0, z_exp_dp0 = model(dp0_img)

		# calc reconstruction loss (dp0 only)

		recon_pie_loss = recon_loss_func(recon_batch_dp0, dp0_img)

		optimizer.zero_grad()
		model.zero_grad()

		recon_pie_loss.backward(retain_graph=True)
		recon_pie_train_loss += recon_pie_loss.data[0].item()

		# calc cosine similarity loss

		sim_loss = cosine_loss_func(z_per_dp0, z_per_dp9, 1) + cosine_loss_func(z_exp_dp0, z_exp_dp1, 1) # similarity
		#dis_loss = cosine_loss_func(z_exp_dp0, z_exp_dp9, -1) + cosine_loss_func(z_per_dp0, z_per_dp1, -1) # dissimilarity

		cosine_train_loss += sim_loss.data[0].item()

		# calc L1 loss

		L1_loss = L1(z_per_dp9, z_per_dp0) + L1(z_exp_dp1, z_exp_dp0)


		# calc triplet loss

		triplet_loss = triplet_loss_func(z_per_dp0, z_per_dp9, z_per_dp1) + triplet_loss_func(z_exp_dp0, z_exp_dp1, z_exp_dp9)
			# triplet(anchor, positive, negative)

		triplet_train_loss += triplet_loss.data[0].item()

		# BCE expression loss

		smile_target = torch.ones(z_exp_dp0.size()).cuda()
		neutral_target = torch.zeros(z_exp_dp0.size()).cuda()

		if dp0_ide == '01': #neutral
			expression_loss = BCE(z_exp_dp0, neutral_target)
		else: #smile
			expression_loss = BCE(z_exp_dp0, smile_target)

		if dp9_ide == '01': #neutral
			expression_loss = expression_loss + BCE(z_exp_dp9, neutral_target)
		else: #smile
			expression_loss = expression_loss + BCE(z_exp_dp9, smile_target)

		if dp1_ide == '01': #neutral
			expression_loss = expression_loss + BCE(z_exp_dp1, neutral_target)
		else: #smile
			expression_loss = expression_loss + BCE(z_exp_dp1, smile_target)

		expression_train_loss += expression_loss[0].item()

		# calc gradients for all losses except swap

		losses = L1_loss + sim_loss + triplet_loss + expression_loss
		losses.backward(retain_graph=True)


		# calc swapping loss

		z_per0_exp9 = torch.cat((z_per_dp0, z_exp_dp9), dim=1) # should be equal to img9 (per0 and per9 are the same)
		recon_per0_exp9 = model.decoder(z_per0_exp9)

		z_per0_exp1 = torch.cat((z_per_dp0, z_exp_dp1), dim=1) # should be equal to img0 (exp1 and exp0 are the same)
		recon_per0_exp1 = model.decoder(z_per0_exp1)

		z_per9_exp0 = torch.cat((z_per_dp9, z_exp_dp0), dim=1) # should be equal to img0
		recon_per9_exp0 = model.decoder(z_per9_exp0)

		z_per1_exp0 = torch.cat((z_per_dp1, z_exp_dp0), dim=1) # should be equal to img1
		recon_per1_exp0 = model.decoder(z_per1_exp0)


		swap_loss1 = recon_loss_func(recon_per0_exp9, dp9_img)
		swap_loss1.backward(retain_graph=True)

		swap_loss2 = recon_loss_func(recon_per0_exp1, dp0_img)
		swap_loss2.backward(retain_graph=True)

		swap_loss3 = recon_loss_func(recon_per9_exp0, dp0_img)
		swap_loss3.backward(retain_graph=True)

		swap_loss4 = recon_loss_func(recon_per1_exp0, dp1_img)
		swap_loss4.backward()

		swap_loss = swap_loss1 + swap_loss2 + swap_loss3 + swap_loss4

		swap_train_loss += swap_loss.data[0].item()

		optimizer.step()

		optimizer.zero_grad()
		model.zero_grad()

		##### MUG #####

		## recon ##

		recon_batch_dp3, z_dp3, z_per_dp3, z_exp_dp3 = model(dp3_img)
		recon_batch_dp4, z_dp4, z_per_dp4, z_exp_dp4 = model(dp4_img)

		recon_batch_dp2, z_dp2, z_per_dp2, z_exp_dp2 = model(dp2_img)


		recon_mug_loss = recon_loss_func(recon_batch_dp2, dp2_img)

		recon_mug_loss.backward(retain_graph=True)
		recon_mug_train_loss += recon_mug_loss.data[0].item()


		### intensity ###

		target = torch.zeros(z_exp_dp2[0].size()).cuda()
		target[int(inten2[0])] = 1.0
		print(target)
		inten_loss = BCE(z_exp_dp2, target)
		inten_loss.backward()
		inten_train_loss += inten_loss.data[0].item()

		optimizer.step()

		print('PIE Train Epoch: {} [{}/{} ({:.0f}%)] Recon: {:.6f} Cosine: {:.6f} Triplet: {:.6f} Swap: {:.6f}'.format(
			epoch, batch_idx * opt.batchSize, (len(dataloader) * opt.batchSize),
			100. * batch_idx / len(dataloader),
			recon_mug_loss.data[0].item(), sim_loss.data[0].item(), triplet_loss.data[0].item(), swap_loss.data[0].item()))
		print('MUG: recon: {:.6f} inten: {:.6f}'.format(recon_mug_loss.data[0].item(), inten_loss.data[0].item()))
			#loss is calculated for each img, so divide by batch size to get loss for the batch

	lossfile.write('PIE Epoch:{} Recon:{:.6f} Swap:{:.6f} ExpLoss:{:.6f}\n'.format(epoch, recon_pie_train_loss,
		swap_train_loss, expression_train_loss))
	lossfile.write('PIE Epoch:{} cosineSim:{:.6f} triplet:{:.6f}\n'.format(epoch, cosine_train_loss,
		triplet_train_loss))
	lossfile.write('MUG Epoch:{} Recon:{:.6f} triplet:{:.6f}\n'.format(epoch, recon_mug_train_loss,
		triplet_train_loss))
	lossfile.write('MUG Epoch:{} Recon:{:.6f} inten:{:.6f}\n'.format(epoch, recon_mug_train_loss,
		inten_train_loss))

	print('====> Epoch: {} Average recon loss: {:.6f} Average cosine loss: {:.6f} Average triplet: {:.6f} Average swap: {:.6f}'.format(
		  epoch, recon_pie_train_loss, cosine_train_loss,
		  triplet_train_loss, swap_train_loss))
			#divide by (batch_size * num_batches) to get average loss for the epoch

	#data
	visualizeAsImages(dp0_img.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_train_img0', n_sample = 18, nrow=5, normalize=False)

	#reconstruction (dp0 only)
	visualizeAsImages(recon_batch_dp0.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_train_recon0', n_sample = 18, nrow=5, normalize=False)

	print('Train data and reconstruction saved.')


	return recon_pie_train_loss / (len(dataloader) * opt.batchSize), triplet_train_loss / (len(dataloader) * opt.batchSize)


def test(epoch):
	print("test")
	model.eval()
	recon_test_loss = 0
	cosine_test_loss = 0
	triplet_test_loss = 0

	pie_dataroot = random.sample(MultipieData,1)[0]
	# mug_dataroot = random.sample(MugData,1)[0]  # why is it without replacement?

	dataset = MUGloader.TrainTestSplit(opt, pieroot=pie_dataroot, mugroot=MugData_root, resize=64)
	print('# size of the current (sub)dataset is %d' %len(dataset))
   # train_amount = train_amount + len(dataset)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
	for batch_idx, data_point in enumerate(dataloader, 0):
		gc.collect() # collect garbage

		dp0_img, dp9_img, dp1_img, dp0_ide, dp9_ide, dp1_ide, dp2_img, dp3_img, dp4_img, inten2, inten3, inten4 = data_point
		dp0_img, dp9_img, dp1_img = parseSampledDataTriplet(dp0_img, dp9_img, dp1_img)
		dp2_img, dp3_img, dp4_img = parseSampledDataTriplet(dp2_img, dp3_img, dp4_img)

		if opt.cuda:
			dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
			dp2_img, dp3_img, dp4_img = setCuda(dp2_img, dp3_img, dp4_img)
		dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img)
		dp2_img, dp3_img, dp4_img = setAsVariable(dp2_img, dp3_img, dp4_img)


		z_recon_dp9, z_dp9, z_per_dp9, z_exp_dp9 = model(dp9_img)
		z_recon_dp1, z_dp1, z_per_dp1, z_exp_dp1 = model(dp1_img)

		optimizer.zero_grad()
		model.zero_grad()

		recon_batch_dp0, z_dp0, z_per_dp0, z_exp_dp0 = model(dp0_img)

		# save test images

		visualizeAsImages(dp0_img.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_img0', n_sample = 18, nrow=5, normalize=False)

		visualizeAsImages(dp9_img.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_img9', n_sample = 18, nrow=5, normalize=False)

		visualizeAsImages(dp1_img.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_img1', n_sample = 18, nrow=5, normalize=False)

		# test disentangling

		z_per0_exp9 = torch.cat((z_per_dp0, z_exp_dp9), dim=1) # should be person 0 with expression 9
		recon_per0_exp9 = model.decoder(z_per0_exp9)

		visualizeAsImages(recon_per0_exp9.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_per0_exp9', n_sample = 18, nrow=5, normalize=False)

		z_per0_exp1 = torch.cat((z_per_dp0, z_exp_dp1), dim=1) # should look the same as dp0_img (exp1 and exp0 are the same)
		recon_per0_exp1 = model.decoder(z_per0_exp1)

		visualizeAsImages(recon_per0_exp1.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_per0_exp1', n_sample = 18, nrow=5, normalize=False)

		z_per1_exp9 = torch.cat((z_per_dp1, z_exp_dp9), dim=1) # should be unique
		recon_per1_exp9 = model.decoder(z_per1_exp9)

		visualizeAsImages(recon_per1_exp9.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_per1_exp9', n_sample = 18, nrow=5, normalize=False)


		# calc reconstruction loss (dp0 only)

		recon_loss = recon_loss_func(recon_batch_dp0, dp0_img)
		optimizer.zero_grad()
		recon_test_loss += recon_loss.data[0].item()

		# calc cosine loss

		sim_loss = cosine_loss_func(z_per_dp0, z_per_dp9, 1) + cosine_loss_func(z_exp_dp0, z_exp_dp1, 1) # similarity

		cosine_test_loss = sim_loss.data[0].item()

		# calc L1 loss

		L1_loss = L1(z_per_dp9, z_per_dp0) + L1(z_exp_dp1, z_exp_dp0)

		# calc triplet loss

		triplet_loss = triplet_loss_func(z_per_dp0, z_per_dp9, z_per_dp1) + triplet_loss_func(z_exp_dp0, z_exp_dp1, z_exp_dp9)
			# triplet(anchor, positive, negative)
		triplet_test_loss = triplet_loss.data[0].item()

		##### MUG #####
		recon_batch_dp3, z_dp3, z_per_dp3, z_exp_dp3 = model(dp3_img)
		recon_batch_dp4, z_dp4, z_per_dp4, z_exp_dp4 = model(dp4_img)

		recon_batch_dp2, z_dp2, z_per_dp2, z_exp_dp2 = model(dp2_img)

		# save test image

		visualizeAsImages(dp2_img.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_img2', n_sample = 18, nrow=5, normalize=False)


		# test intensity interpolation

		img_list = []
		for i in range(11):
			z_exp_test = torch.full(z_exp_dp2.size(), i / 10)
			z_test = torch.cat((z_per_dp2.cuda(), z_exp_test.cuda()), dim=1)
			recon_test = model.decoder(z_test)
			#img_list.append(recon_test)
			vutils.save_image(recon_test, os.path.join(opt.dirImageoutput, 'e_'+str(epoch)+'_intentest' + str(i) + '.jpg'))

		# img_list = torch.stack(img_list, 0)
		# images = img_list[0:11,:,:,:]
		# vutils.save_image(images, os.path.join(opt.dirImageoutput, 'e_'+str(epoch)+'_test_inten_img2'), nrow=2)

	print('Test images saved')
	print('====> Test set recon loss: {:.4f}\ttriplet loss:  {:.4f}'.format(recon_test_loss, triplet_test_loss))


def load_last_model():
	 models = glob(opt.dirCheckpoints + '/*.pth')
	 model_ids = [(int(f.split('_')[1]), f) for f in models]
	 start_epoch, last_cp = max(model_ids, key=lambda item:item[0])  # max returns the model_id with the largest proxy value (item)
	 model.load_state_dict(torch.load(last_cp))
	 return start_epoch, last_cp

def start_training():
	# start_epoch, _ = load_last_model()
	start_epoch = 0
	#test(start_epoch)

	for epoch in range(start_epoch + 1, start_epoch + opt.epoch_iter + 1):
		recon_loss, triplet_loss = train(epoch)
		torch.save(model.state_dict(), opt.dirCheckpoints + 'densetripletmug.pth')
		if epoch % 10 == 0:
			test(epoch)

	lossfile.close()

def last_model_to_cpu():
	_, last_cp = load_last_model()
	model.cpu()
	torch.save(model.state_dict(), opt.dirCheckpoints + '/cpu_'+last_cp.split('/')[-1])

if __name__ == '__main__':
	start_training()
	# last_model_to_cpu()
	# load_last_model()
	# rand_faces(10)
	# da = load_pickle(test_loader[0])
	# da = da[:120]
	# it = iter(da)
	# l = zip(it, it, it)
	# # latent_space_transition(l)
	# perform_latent_space_arithmatics(l)
