import os
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter, Sequential, Linear, BatchNorm1d
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_add_pool, global_mean_pool
# "SGConv, GATConv." We can add more models, however, our focuse is on GCN, GAT and GSAGE

class Encoder(torch.nn.Module):
	"""
	We adapt part of the code from https://github.com/divelab/DIG.
	
	Args:
		feat_dim (int): The dimension of input node features.
		hidden_dim (int): The dimension of node-level (local) embeddings. 
		n_layers (int, optional): The number of GNN layers in the encoder. (default: :obj:`5`)
		pool (string, optional): The global pooling methods, :obj:`sum` or :obj:`mean`.
			(default: :obj:`sum`)
		gnn (string, optional): The type of GNN layer, :obj:`gcn` or :obj:`gin` or :obj:`gat`
			or :obj:`graphsage` or :obj:`resgcn` or :obj:`sgc`. (default: :obj:`gcn`)
		bn (bool, optional): Whether to include batch normalization. (default: :obj:`True`)
		node_level (bool, optional): If :obj:`True`, the encoder will output node level
			embedding (local representations). (default: :obj:`False`)
		graph_level (bool, optional): If :obj:`True`, the encoder will output graph level
			embeddings (global representations). (default: :obj:`True`)
		edge_weight (bool, optional): Only applied to GCN. Whether to use edge weight to
			compute the aggregation. (default: :obj:`False`)
	"""

	def __init__(self, feat_dim, hidden_dim, n_layers=5, pool="sum", 
				 gnn="gcn", bn=True, node_level=False, graph_level=True):
		super(Encoder, self).__init__()

		if gnn == "gcn":
			self.encoder = GCN(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "graphsage":
			self.encoder = GraphSAGE(feat_dim, hidden_dim, n_layers, pool, bn)
		elif gnn == "gin":
			self.encoder = GIN(feat_dim, hidden_dim, n_layers, pool, bn)

		self.node_level = node_level
		self.graph_level = graph_level

	def forward(self, data):
		z_g, z_n = self.encoder(data)
		if self.node_level and self.graph_level:
			return z_g, z_n
		elif self.graph_level:
			return z_g
		else:
			return z_n

	def save_checkpoint(self, save_path, optimizer, epoch, best_train_loss, best_val_loss, is_best):
		ckpt = {}
		ckpt["state"] = self.state_dict()
		ckpt["epoch"] = epoch
		ckpt["optimizer_state"] = optimizer.state_dict()
		ckpt["best_train_loss"] = best_train_loss
		ckpt["best_val_loss"] = best_val_loss
		torch.save(ckpt, os.path.join(save_path, "model.ckpt"))
		if is_best:
			torch.save(ckpt, os.path.join(save_path, "best_model.ckpt"))

	def load_checkpoint(self, load_path, optimizer):
		ckpt = torch.load(os.path.join(load_path, "best_model.ckpt"))
		self.load_state_dict(ckpt["state"])
		epoch = ckpt["epoch"]
		best_train_loss = ckpt["best_train_loss"]
		best_val_loss = ckpt["best_val_loss"]
		optimizer.load_state_dict(ckpt["optimizer_state"])
		return epoch, best_train_loss, best_val_loss


class GCN(torch.nn.Module):
	"""
	For a thorough understanding, visit:
	Semi-supervised Classification with Graph Convolutional Networks <https://arxiv.org/abs/1609.02907>.
	"""

	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum", bn=False, xavier=True):
		super(GCN, self).__init__()

		if bn:
			self.bns = torch.nn.ModuleList()
		self.convs = torch.nn.ModuleList()
		self.acts = torch.nn.ModuleList()
		self.n_layers = n_layers
		self.pool = pool

		a = torch.nn.ReLU()

		for i in range(n_layers):
			start_dim = hidden_dim if i else feat_dim
			conv = GCNConv(start_dim, hidden_dim)
			
			if xavier:
				self.weights_init(conv)
				self.convs.append(conv)
				self.acts.append(a)
				
				if bn:
					self.bns.append(BatchNorm1d(hidden_dim))

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, GCNConv):
				layer = m.lin
			if isinstance(m, Linear):
				layer = m
			torch.nn.init.xavier_uniform_(layer.weight.data)
			if layer.bias is not None:
				layer.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data
		xs = []
		for i in range(self.n_layers):
			x = self.convs[i](x, edge_index)
			# Only uncomment for end-to-end task. 
			# x = F.relu(x)
			# x = F.dropout(x, p=0.4, training=self.training)
			x = self.acts[i](x)
			if self.bns is not None:
				x = self.bns[i](x)
			xs.append(x)

		if self.pool == "sum":
			xpool = [global_add_pool(x, batch) for x in xs]
		else:
			xpool = [global_mean_pool(x, batch) for x in xs]
		global_rep = torch.cat(xpool, 1)

		return global_rep, x


class GraphSAGE(torch.nn.Module):
	"""
	For a thorough understanding, visit:
	Inductive Representation Learning on Large Graphs” <https://arxiv.org/abs/1706.02216>.
	"""

	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum", bn=False, xavier=True):
		super(GraphSAGE, self).__init__()

		if bn:
			self.bns = torch.nn.ModuleList()
		self.convs = torch.nn.ModuleList()
		self.acts = torch.nn.ModuleList()
		self.n_layers = n_layers
		self.pool = pool

		a = torch.nn.ReLU()

		for i in range(n_layers):
			start_dim = hidden_dim if i else feat_dim
			conv = SAGEConv(start_dim, hidden_dim)
			if xavier:
				self.weights_init(conv)
			self.convs.append(conv)
			self.acts.append(a)
			if bn:
				self.bns.append(BatchNorm1d(hidden_dim))

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, SAGEConv):
				layers = [m.lin_l, m.lin_r]
			if isinstance(m, Linear):
				layers = [m]
			for layer in layers:
				torch.nn.init.xavier_uniform_(layer.weight.data)
				if layer.bias is not None:
					layer.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data
		xs = []
		for i in range(self.n_layers):
			x = self.convs[i](x, edge_index)
			# Only uncomment for end-to-end task.
			# x = F.relu(x)
			# x = F.dropout(x, p=0.4, training=self.training)
			x = self.acts[i](x)
			if self.bns is not None:
				x = self.bns[i](x)
			xs.append(x)

		if self.pool == "sum":
			xpool = [global_add_pool(x, batch) for x in xs]
		else:
			xpool = [global_mean_pool(x, batch) for x in xs]
		global_rep = torch.cat(xpool, 1)

		return global_rep, x


class GIN(torch.nn.Module):
	"""
	For a thorough understanding, visit:
	How Powerful are Graph Neural Networks? <https://arxiv.org/abs/1810.00826>`.
	"""

	def __init__(self, feat_dim, hidden_dim, n_layers=3, pool="sum", bn=False, xavier=True):
		super(GIN, self).__init__()

		if bn:
			self.bns = torch.nn.ModuleList()
		self.convs = torch.nn.ModuleList()
		self.n_layers = n_layers
		self.pool = pool

		self.act = torch.nn.ReLU()

		for i in range(n_layers):
			start_dim = hidden_dim if i else feat_dim
			mlp = Sequential(Linear(start_dim, hidden_dim),
							self.act,
							Linear(hidden_dim, hidden_dim))
			if xavier:
				self.weights_init(mlp)
			conv = GINConv(mlp)
			self.convs.append(conv)
			if bn:
				self.bns.append(BatchNorm1d(hidden_dim))

	def weights_init(self, module):
		for m in module.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, data):
		x, edge_index, batch = data
		xs = []
		for i in range(self.n_layers):
			x = self.convs[i](x, edge_index)
			# Only uncomment for end-to-end task.
			# x = F.relu(x)
			# x = F.dropout(x, p=0.4, training=self.training)
			x = self.act(x)
			if self.bns is not None:
				x = self.bns[i](x)
			xs.append(x)

		if self.pool == "sum":
			xpool = [global_add_pool(x, batch) for x in xs]
		else:
			xpool = [global_mean_pool(x, batch) for x in xs]
		global_rep = torch.cat(xpool, 1)

		return global_rep, x



class GraphClassificationModel(nn.Module):
	"""
	Model for graph classification.
	Encoder followed by linear layer.
	"""

	def __init__(self, feat_dim, hidden_dim, n_layers, output_dim, gnn, load=None):
		super(GraphClassificationModel, self).__init__()

		self.encoder = Encoder(feat_dim, hidden_dim, n_layers=n_layers, gnn=gnn)

		if load:
			ckpt = torch.load(os.path.join("ckpt", load, "best_model.ckpt"))
			self.encoder.load_state_dict(ckpt["state"])
			for param in self.encoder.parameters():
				param.requires_grad = False

		if gnn in ["resgcn", "sgc"]:
			feat_dim = hidden_dim
		else:
			feat_dim = n_layers * hidden_dim
		self.classifier = nn.Linear(feat_dim, output_dim)

	def forward(self, data):
		embeddings = self.encoder(data)
		scores = self.classifier(embeddings)
		return scores

	def save_checkpoint(self, save_path, optimizer, epoch, best_train_loss, best_val_loss, is_best):
		ckpt = {}
		ckpt["state"] = self.state_dict()
		ckpt["epoch"] = epoch
		ckpt["optimizer_state"] = optimizer.state_dict()
		ckpt["best_train_loss"] = best_train_loss
		ckpt["best_val_loss"] = best_val_loss
		torch.save(ckpt, os.path.join(save_path, "pred_model.ckpt"))
		if is_best:
			torch.save(ckpt, os.path.join(save_path, "best_pred_model.ckpt"))

	def load_checkpoint(self, load_path, optimizer):
		ckpt = torch.load(os.path.join(load_path, "best_pred_model.ckpt"))
		self.load_state_dict(ckpt["state"])
		epoch = ckpt["epoch"]
		best_train_loss = ckpt["best_train_loss"]
		best_val_loss = ckpt["best_val_loss"]
		optimizer.load_state_dict(ckpt["optimizer_state"])
		return epoch, best_train_loss, best_val_loss
		
		
		
		
		
