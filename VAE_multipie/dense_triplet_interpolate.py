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
import MultipieLoader
import gc
import WaspNet_dany as WaspNet

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
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
parser.add_argument('--zdim', type=int, default=128, help='latent variable size')
parser.add_argument('--edim', type=int, default=64, help='dimensions of expression vec')
parser.add_argument('--pdim', type=int, default=64, help='dimensions of person vec')

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


def parseSampledDataTripletMultipie(dp0_img,  dp9_img, dp1_img):
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



# Training data folder list
Data = []
#session 01

Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_01_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_02_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_03_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_04_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_05_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_06_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_07_select/')

'''
#session 02
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_01_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_02_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_03_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_04_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_05_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_06_select/')
Data.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_07_select/')
#session 03
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_05_select/')
#session 04
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_05_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_06_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_07_select/')
'''

# Small Testing Set
# TestingData = []
# TestingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_select_test/')

class AE(nn.Module):
	def __init__(self, latent_variable_size):
		super(AE, self).__init__()
		# self.latent_variable_size = latent_variable_size

		self.encoder = WaspNet.Dense_Encoders_AE_SliceSplit(opt)
		self.decoder = WaspNet.Dense_Decoders_AE(opt)

	def forward(self, x):
		z, z_per, z_exp = self.encoder(x)
		recon_x = self.decoder(z)
		return recon_x, z, z_per, z_exp

model=AE(latent_variable_size=128)

model.load_state_dict(torch.load('Epoch_999_Recon_0.0000_cosine_0.0000.pth')) # pretrained encoder and decoder


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


def test(epoch):
	print("test")
	model.eval()
	recon_test_loss = 0
	cosine_test_loss = 0
	triplet_test_loss = 0
	dataroot = random.sample(Data,1)[0]

	dataset = MultipieLoader.FareMultipieExpressionTripletsFrontalTrainTestSplit(opt, root=dataroot, resize=64)
	print('# size of the current (sub)dataset is %d' %len(dataset))
   # train_amount = train_amount + len(dataset)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
	for batch_idx, data_point in enumerate(dataloader, 0):
		gc.collect() # collect garbage

		dp0_img, dp9_img, dp1_img, dp0_ide, dp9_ide, dp1_ide = data_point
		dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
		if opt.cuda:
			dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
		dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )


		z_dp9, z_per_dp9, z_exp_dp9 = model.get_latent_vectors(dp9_img)
		z_dp1, z_per_dp1, z_exp_dp1 = model.get_latent_vectors(dp1_img)

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
		recon_per0_exp9 = model.decode(z_per0_exp9)

		visualizeAsImages(recon_per0_exp9.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_per0_exp9', n_sample = 18, nrow=5, normalize=False)

		z_per0_exp1 = torch.cat((z_per_dp0, z_exp_dp1), dim=1) # should look the same as dp0_img (exp1 and exp0 are the same)
		recon_per0_exp1 = model.decode(z_per0_exp1)

		visualizeAsImages(recon_per0_exp1.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_per0_exp1', n_sample = 18, nrow=5, normalize=False)

		z_per1_exp9 = torch.cat((z_per_dp1, z_exp_dp9), dim=1) # should be unique
		recon_per1_exp9 = model.decode(z_per1_exp9)

		visualizeAsImages(recon_per1_exp9.data.clone(),
		opt.dirImageoutput,
		filename='e_'+str(epoch)+'_test_per1_exp9', n_sample = 18, nrow=5, normalize=False)

		# test interpolatation

		exp_diff = z_exp_dp9 - z_exp_dp0
		img_list = []
		for i in range(11):
			z_exp_test = z_exp_dp0 + i * (exp_diff / 10)
			z_test = torch.cat((z_per_dp0.cuda(), z_exp_test.cuda()), dim=1)
			recon_test = model.decode(z_test)
			#img_list.append(recon_test)
			vutils.save_image(recon_test, os.path.join(opt.dirImageoutput, 'e_'+str(epoch)+'_intentest' + str(i) + '.jpg'))



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


	for epoch in range(start_epoch + 1, start_epoch + opt.epoch_iter + 1):
		test(epoch)

	lossfile.close()

def last_model_to_cpu():
	_, last_cp = load_last_model()
	model.cpu()
	torch.save(model.state_dict(), opt.dirCheckpoints + '/cpu_'+last_cp.split('/')[-1])

if __name__ == '__main__':
	for epoch in range(1, opt.epoch_iter + 1):
		if epoch % 10 == 0 or epoch == 1:
			test(epoch)
	# last_model_to_cpu()
	# load_last_model()
	# rand_faces(10)
	# da = load_pickle(test_loader[0])
	# da = da[:120]
	# it = iter(da)
	# l = zip(it, it, it)
	# # latent_space_transition(l)
	# perform_latent_space_arithmatics(l)
