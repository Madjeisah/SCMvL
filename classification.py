import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from data_loader import *
from net_model import *
from hparams import Classi_Hparams

hparams = Classi_Hparams()

# set random seed
SEED = 0
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)  # Numpy module.
random.seed(SEED)  # Python random module.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Params:

	def __init__(self):
	
		self.parser = hparams.parser
		self.parse()
		self.save_dir()

	def parse(self):
		self.prm = self.parser.parse_args()
		
	def save_dir(self):
		if not os.path.isdir(os.path.join("executes", self.prm.save)):
			os.makedirs(os.path.join("executes", self.prm.save))
		if not os.path.isdir(os.path.join("ckpt", self.prm.save)):
			os.makedirs(os.path.join("ckpt", self.prm.save))

	def __str__(self):
		return ("Classification setup:\n" + "".join(["-"] * 45) + "\n" + "\n".join(["{:<18} |-------> {}".format(k, v) for k, v in vars(self.prm).items()]) + "\n" + "".join(["-"] * 45) + "\n")


def run(args, epochs, mode, dataloader, model, optimizer):
	if mode == "train":
		model.train()
	elif mode == "val" or mode == "test":
		model.eval()
	else:
		assert False, "Wrong Mode:{} for Run".format(mode)

	# CrossEntropy loss for classification task
	loss_fn = torch.nn.CrossEntropyLoss()

	losses = []
	correct = 0
	with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epochs)) as t:
		for data in dataloader:
			data.to(device)

			data_input = data.x, data.edge_index, data.batch
			labels = data.y

			# get class scores from model
			scores = model(data_input)

			# compute cross entropy loss
			loss = loss_fn(scores, labels)

			if mode == "train":
				# backprop
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			# keep track of loss and accuracy
			pred = scores.argmax(dim=1)
			correct += int((pred == labels).sum())
			losses.append(loss.item())
			t.set_postfix(loss=losses[-1])
			t.update()

	# gather the results for the epoch
	epoch_loss = sum(losses) / len(losses)
	accuracy = correct / len(dataloader.dataset)
	return epoch_loss, accuracy


def main(args):
	dataset, input_dim, num_classes = load_dataset(args.dataset)

	# split the data into train / val / test sets
	train_dataset, val_dataset, test_dataset = split_dataset(dataset, args.train_data_percent)

	# build_classification_loader is a dataloader which gives one graph at a time
	train_loader = build_classification_loader(args, train_dataset, "train")
	val_loader = build_classification_loader(args, val_dataset, "val")
	test_loader = build_classification_loader(args, test_dataset, "test")

	print("Dataset Split: {} {} {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
	print("Number of Classes: {}".format(num_classes))

	# classification model is a GNN encoder followed by linear layer
	model = GraphClassificationModel(input_dim, args.feat_dim, n_layers=args.layers, output_dim=num_classes, gnn=args.model, load=args.load)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	best_train_loss, best_val_loss = float("inf"), float("inf")
	logger = SummaryWriter(os.path.join("executes", args.save))

	best_valid_epoch = 0
	test_accs = []
	for r in range(1, args.runs):
		print('')
		print(f'Run {r:02d}:')
		print('')

		print('Training Initialization...')
		print('')

		for epoch in range(args.epochs):
			train_loss, train_acc = run(args, epoch, "train", train_loader, model, optimizer)
			print("Train Epoch Loss: {}, Accuracy: {}".format(train_loss, train_acc))
			logger.add_scalar("Train Loss", train_loss, epoch)

			val_loss, val_acc = run(args, epoch, "val", val_loader, model, optimizer)
			print("Val Epoch Loss: {}, Accuracy: {}".format(val_loss,val_acc))
			logger.add_scalar("Val Loss", val_loss, epoch)
			
			# save best model 
			is_best_loss = False
			if val_loss < best_val_loss:
				best_epoch, best_train_loss, best_val_loss, is_best_loss = epoch, train_loss, val_loss, True
				best_valid_epoch = epoch
			model.save_checkpoint(os.path.join("ckpt", args.save), optimizer, epoch, best_train_loss, best_val_loss, is_best_loss)
			
		# load best model for evaluation
		best_epoch, best_train_loss, best_val_loss = model.load_checkpoint(os.path.join("ckpt", args.save), optimizer)
		model.eval()
		
		t_loss, t_accuracy = run(args, best_epoch, "test", test_loader, model, optimizer)
		print("Test Epoch Loss: {}, Accuracy: {}".format(t_loss, t_accuracy))
		test_accs.append(t_accuracy)

	test_acc = torch.tensor(test_accs)
	print('')
	print('=====================================')
	print(f'Overall Test Accuracy: {test_acc.mean():.4f} ± {test_acc.std():.3f}') 
	print('=====================================')    
	print('')
	print('')

	sys.stdout.flush()

if __name__ == "__main__":

	args = Params()
	print(args)

	main(args.prm)
