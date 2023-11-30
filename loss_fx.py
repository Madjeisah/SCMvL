import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class infonce_GCC(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class infonce_GCC_NS(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        label = torch.arange(bsz).cuda().long()
        loss = self.criterion(x, label)
        return loss
        
        
class infonce(nn.Module):
	"""
	The InfoNCE (NT-XENT) loss in contrastive learning. The implementation
	follows the paper `A Simple Framework for Contrastive Learning of 
	Visual Representations <https://arxiv.org/abs/2002.05709>`.
	"""

	def __init__(self):
		super(infonce, self).__init__()

		self.tau = 0.5
		self.norm = True

	def forward(self, embed_anchor, embed_positive):
		"""
		Args:
			embed_anchor, embed_positive: Tensor of shape [batch_size, embed_dim]
			tau: Float. Usually in (0,1].
			norm: Boolean. Whether to apply normalization.
		"""

		batch_size = embed_anchor.shape[0]
		sim_matrix = torch.einsum("ik,jk->ij", embed_anchor, embed_positive)

		if self.norm:
			embed_anchor_abs = embed_anchor.norm(dim=1)
			embed_positive_abs = embed_positive.norm(dim=1)
			sim_matrix = sim_matrix / torch.einsum("i,j->ij", embed_anchor_abs, embed_positive_abs)

		sim_matrix = torch.exp(sim_matrix / self.tau)
		pos_sim = sim_matrix[range(batch_size), range(batch_size)]
		loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
		loss = - torch.log(loss).mean()
		return loss
