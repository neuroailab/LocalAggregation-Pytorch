"""
Non-parametric Instance Discrimination Loss
https://github.com/zhirongw/lemniscate.pytorch

Code is based on Tensorflow implementation: 
https://github.com/neuroailab/LocalAggregation

This script wraps the InstanceDiscriminationLoss function as a torch.nn.Module,
so that the loss can be computed parallelly across multi-gpus using Dataparallel

"""
import math
import torch
import numpy as np

from src.utils.tensor import l2_normalize


class InstanceDiscriminationLossModule(torch.nn.Module):
    def __init__(self, memory_bank_broadcast, cluster_labels_broadcast=None, k=4096, t=0.07, m=0.5):
        super(InstanceDiscriminationLossModule, self).__init__()
        self.k, self.t, self.m = k, t, m

        self.indices = None
        self.outputs = None
        self._bank = None  # pass in via forward function
        self.memory_bank_broadcast = memory_bank_broadcast
        self.data_len = memory_bank_broadcast[0].size(0)

    def _softmax(self, dot_prods):
        Z = 2876934.2 / 1281167 * self.data_len
        return torch.exp(dot_prods / self.t) / Z

    def updated_new_data_memory(self, indices, outputs):
        outputs = l2_normalize(outputs)
        data_memory = torch.index_select(self._bank, 0, indices)
        new_data_memory = data_memory * self.m + (1 - self.m) * outputs
        return l2_normalize(new_data_memory, dim=1)

    def synchronization_check(self):
        for i in range(len(self.memory_bank_broadcast)):
            if i == 0:
                device = self.memory_bank_broadcast[0].device
            else:
                assert torch.equal(self.memory_bank_broadcast[0], self.memory_bank_broadcast[i].to(device))

    def _get_dot_products(self, vec, idxs):
        """
        This function is copied from the get_dot_products in Memory_Bank class
        Since we want to register self._bank as a buffer (to be broadcasted to multigpus) instead of self.memory_bank,
        we need to avoid calling self.memory_bank get_dot_products

        """
        vec_shape = list(vec.size())  # [bs, dim]
        idxs_shape = list(idxs.size())  # [bs, ...]

        assert len(idxs_shape) in [1, 2]
        assert len(vec_shape) == 2
        assert vec_shape[0] == idxs_shape[0]

        if len(idxs_shape) == 1:
            with torch.no_grad():
                memory_vecs = torch.index_select(self._bank, 0, idxs)
                memory_vecs_shape = list(memory_vecs.size())
                assert memory_vecs_shape[0] == idxs_shape[0]
        else:  # len(idxs_shape) == 2
            with torch.no_grad():
                batch_size, k_dim = idxs.size(0), idxs.size(1)
                flat_idxs = idxs.view(-1)
                memory_vecs = torch.index_select(self._bank, 0, flat_idxs)
                memory_vecs = memory_vecs.view(batch_size, k_dim, self._bank.size(1))
                memory_vecs_shape = list(memory_vecs.size())

            vec_shape[1:1] = [1] * (len(idxs_shape) - 1)
            vec = vec.view(vec_shape)  # [bs, 1, dim]

        prods = memory_vecs * vec
        assert list(prods.size()) == memory_vecs_shape

        return torch.sum(prods, dim=-1)

    def compute_data_prob(self):
        logits = self._get_dot_products(self.outputs, self.indices)
        return self._softmax(logits)

    def compute_noise_prob(self):
        batch_size = self.indices.size(0)
        noise_indx = torch.randint(0, self.data_len, (batch_size, self.k),
                                   device=self.outputs.device)  # U(0, data_len)
        noise_indx = noise_indx.long()
        logits = self._get_dot_products(self.outputs, noise_indx)
        noise_probs = self._softmax(logits)
        return noise_probs

    def forward(self, indices, outputs, gpu_idx):
        self.indices = indices.detach()
        self.outputs = l2_normalize(outputs, dim=1)
        self._bank = self.memory_bank_broadcast[gpu_idx]

        batch_size = self.indices.size(0)
        data_prob = self.compute_data_prob()
        noise_prob = self.compute_noise_prob()

        assert data_prob.size(0) == batch_size
        assert noise_prob.size(0) == batch_size
        assert noise_prob.size(1) == self.k

        base_prob = 1.0 / self.data_len
        eps = 1e-7

        ## Pmt
        data_div = data_prob + (self.k * base_prob + eps)

        ln_data = torch.log(data_prob) - torch.log(data_div)

        ## Pon
        noise_div = noise_prob + (self.k * base_prob + eps)
        ln_noise = math.log(self.k * base_prob) - torch.log(noise_div)

        curr_loss = -(torch.sum(ln_data) + torch.sum(ln_noise))
        curr_loss = curr_loss / batch_size

        new_data_memory = self.updated_new_data_memory(self.indices, self.outputs)

        return curr_loss.unsqueeze(0), new_data_memory

