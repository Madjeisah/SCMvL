import os.path as osp
import torch
import numpy as np
import torch.nn as nn
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from data_loader import *
from loss_fx import *
from net_model import *
from hparams import Hparams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hparams = Hparams()

def set_seed(seed):
	"""
	Utility function to set seed values for RNG for various modules
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

class Params:

	def __init__(self):
	
		self.parser = hparams.parser
		self.parse()
		self.save_to()

	def parse(self):
		self.prm = self.parser.parse_args()

	def save_to(self):
		if not os.path.isdir(os.path.join("executes", self.prm.save)):
			os.makedirs(os.path.join("executes", self.prm.save))
		if not os.path.isdir(os.path.join("ckpt", self.prm.save)):
			os.makedirs(os.path.join("ckpt", self.prm.save))

	def __str__(self):
		return ("Pre-train setup:\n" + "".join(["-"] * 50) + "\n" + "\n".join(["{:<18} -------> {}".format(k, v) for k, v in vars(self.prm).items()]) + "\n" + "".join(["-"] * 50) + "\n")

def run(args, epoch, mode, dataloader, model, optimizer):
	if mode == "train":
		model.train()
	elif mode == "val" or mode == "test":
		model.eval()
	else:
		assert False, "Wrong Mode:{} for Execution".format(mode)

	losses = []
	contrastive_fn = eval(args.loss + "()")
	with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
		for data in dataloader:
			data.to(device)

			# readout_anchor is the embedding of the original datapoint x on passing through the model
			readout_anchor = model((data.x_anchor, data.edge_index_anchor, data.x_anchor_batch))

			# readout_positive is the embedding of the positively augmented x on passing through the model
			readout_positive = model((data.x_pos, data.edge_index_pos, data.x_pos_batch))

			# negative samples for calculating the contrastive loss is computed in contrastive_fn
			loss = contrastive_fn(readout_anchor, readout_positive)

			if mode == "train":
				# backprop
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			# keep track of loss values
			losses.append(loss.item())
			t.set_postfix(loss=losses[-1])
			t.update()

	# gather the results for the epoch
	epoch_loss = sum(losses) / len(losses)
	return epoch_loss

def main(args):
	dataset, input_dim, num_classes = load_dataset(args.dataset)

	# split the data into train / val / test sets
	train_dataset, val_dataset, test_dataset = split_dataset(dataset, args.train_data_percent)

	# build_loader is a dataloader which gives a paired sampled - the original x and the positively
	# augmented x obtained by applying the transformations in the augment_list as an argument
	train_loader = build_loader(args, train_dataset, "train")
	val_loader = build_loader(args, val_dataset, "val")
	test_loader = build_loader(args, test_dataset, "test")

	# easy initialization of the GNN model encoder to map graphs to embeddings needed for contrastive training
	model = Encoder(input_dim, args.feat_dim, n_layers=args.layers, gnn=args.model)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)

	best_train_loss, best_val_loss = float("inf"), float("inf")

	logger = SummaryWriter(os.path.join("executes", args.save))

	train_losses, val_losses = [], []

	for epoch in range(args.epochs):
		train_loss = run(args, epoch, "train", train_loader, model, optimizer)
		print("Train Epoch Loss: {:.3f}".format(train_loss))
		logger.add_scalar("Train Loss", train_loss, epoch)
		train_losses.append(train_loss)

		val_loss = run(args, epoch, "val", val_loader, model, optimizer)
		print("Val Epoch Loss: {:.3f}".format(val_loss))
		logger.add_scalar("Val Loss", val_loss, epoch)
		val_losses.append(val_loss)

		# save model
		is_best_loss = False
		if val_loss < best_val_loss:
			best_epoch, best_train_loss, best_val_loss, is_best_loss = epoch, train_loss, val_loss, True

		model.save_checkpoint(os.path.join("ckpt", args.save), optimizer, epoch, best_train_loss, best_val_loss, is_best_loss)

	print("Train Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_train_loss))
	print("Val Loss at epoch {} (best model): {:.3f}".format(best_epoch, best_val_loss))

	best_epoch, best_train_loss, best_val_loss = model.load_checkpoint(os.path.join("ckpt", args.save), optimizer)
	model.eval()

	test_loss = run(args, best_epoch, "test", test_loader, model, optimizer)
	print("Test Loss at epoch {}: {:.3f}".format(best_epoch, test_loss))

if __name__ == "__main__":

	set_seed(0)
	args = Params()
	print(args)

	main(args.prm)
