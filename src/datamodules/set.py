from .base import BaseStreamingDataset, BaseStreamingPLDataModule
from typing import Optional, Callable, Dict
from src.utils.datamodule import select_distribution
import random
import numpy as np
import hydra
import torch
from tqdm import tqdm
import omegaconf
from omegaconf import DictConfig, OmegaConf

class SetDataset(BaseStreamingDataset):
    def __init__(self, dataset_parameters, **kwargs):
        super().__init__(dataset_parameters, **kwargs)
        self.batch_size = dataset_parameters.batch_size
        self.num_batches = kwargs['num_batches']
        self.seed = dataset_parameters.seed
        self.seq_min_length = kwargs['seq_min_length']
        self.seq_max_length = kwargs['seq_max_length']
        self.architecture = kwargs["mixing_architecture"]["name"]
        self.architecture_config = kwargs["mixing_architecture"]
        self.split = kwargs['split']
        assert self.seq_min_length <= self.seq_max_length, f"seq_min_length ({self.seq_min_length}) should be less than or equal to seq_max_length ({self.seq_max_length})"
        self.x_dim = dataset_parameters.x_dim
        self.y_dim = dataset_parameters.y_dim
        self.phi_dim = dataset_parameters.phi_dim
        self.distribution: Callable = select_distribution(kwargs["distribution_config"])
        self.generation_mode = kwargs.get("generation_mode", "offline")
        self.use_constraints = kwargs.get("use_constraints", False)
        self.constraints = kwargs.get("constraints", None)
        random.seed(self.seed)
        
        # We should use the same mixing function for all splits
        if kwargs.get("mixing_fn", None) is not None:
            self.mixing_fn = kwargs["mixing_fn"]
        else:
            self.mixing_fn = self.mixing(kwargs["mixing_architecture"])

        self.mixing_type = kwargs["mixing_type"]
        
    def generate_batch(self):


        def sample_with_constraints(constraints, lengths, num_batches, batch_size, seq_max_length, x_dim):
            counter = 0
            reserve_n = 5
            default_low = constraints["default_low"]
            default_high = constraints["default_high"]
            lbs = constraints["lb"]
            hbs = constraints["hb"]
            rejection_sampling = constraints.get("rejection_sampling", False)
            fraction = constraints.get("fraction", 1.0)
            use_fraction = constraints.get("use_fraction", False)

            lb = [torch.zeros(x_dim) + default_low for i in range(len(lbs))]
            hb = [torch.zeros(x_dim) + default_high for i in range(len(hbs))]

            if not use_fraction: # use the explicit dimensions provided in the constraints
                for i, lb_ in enumerate(lbs):
                    if isinstance(lb_, omegaconf.dictconfig.DictConfig) or isinstance(lb_, dict):
                        for key, value in lb_.items():
                            lb[i][int(key)] = value
                    else:
                        lb[i] = torch.zeros(x_dim) + lb_
                        
                for i, hb_ in enumerate(hbs):
                    if isinstance(hb_, omegaconf.dictconfig.DictConfig) or isinstance(hb_, dict):
                        for key, value in hb_.items():
                            hb[i][int(key)] = value
                    else:
                        hb[i] = torch.zeros(x_dim) + hb_
            else: # sample the dimensions randomly to be used with the constraints
                for i in range(len(lbs)):
                    dims = random.sample(range(x_dim), int(fraction * x_dim))
                    for dim in dims:
                        lb[i][dim] = lbs[i]
                        hb[i][dim] = hbs[i]

            sets = []

            if rejection_sampling:

                # we have to sample all dimensions from the complete distribution
                sets_ = torch.rand(x_dim * num_batches * batch_size * reserve_n, seq_max_length)
                # as long as there aren't num_batches * batch_size samples per dimension that satisfy the condition, resample the rest
                # such that at least num_batches * batch_size samples satisfy the condition per dimension
                # first we need to expand each lb[i] such that it can be used for comparison with sets_
                lb = [lb_.expand(num_batches * batch_size * reserve_n, x_dim).permute(1, 0).reshape(-1) for lb_ in lb] # each lb will be num_batches * batch_size * x_dim * 10 * seq_max_length
                # and each num_batches * batch_size * 10 * seq_max_length consecutive samples will correspond to one dimension out of x_dim
                hb = [hb_.expand(num_batches * batch_size * reserve_n, x_dim).permute(1, 0).reshape(-1) for hb_ in hb]

                # now, as long as there aren't num_batches * batch_size samples that satisfy the condition, resample the rest
                # such that at least num_batches * batch_size samples satisfy the condition
                masks_tensorlist = [((sets_.sum(-1) >= lb_) & (sets_.sum(-1) <= hb_)) for lb_, hb_ in zip(lb, hb)]
                mask = torch.stack(masks_tensorlist, dim=-1).any(dim=-1) # logical or among the conditions (for multi-region constraints) # x_dim * num_batches * batch_size * 10
                while not (mask.reshape(x_dim, -1).sum(1) >= num_batches * batch_size).all():
                    resample_indices = ~mask
                    # resample_indices = (~(mask.reshape(x_dim, -1).sum(1) >= num_batches * batch_size)).expand(num_batches * batch_size * reserve_n, x_dim).permute(1,0).reshape(-1)
                    sets_[resample_indices, :] = torch.rand_like(sets_[resample_indices, :])
                    masks_tensorlist = [(sets_.sum(-1) >= lb_) & (sets_.sum(-1) <= hb_) for lb_, hb_ in zip(lb, hb)]
                    mask = torch.stack(masks_tensorlist, dim=-1).any(dim=-1) # num_batches * batch_size * x_dim * 10

                indices = torch.stack([torch.where(mask.reshape(x_dim, -1)[i, :])[0][:num_batches * batch_size] for i in range(x_dim)], dim=0) # x_dim, num_batches * batch_size
                sets = torch.cat([sets_[i * (num_batches * batch_size * reserve_n): (i + 1) * (num_batches * batch_size * reserve_n), :][indices[i], :] for i in range(x_dim)], dim=-1).reshape(-1, x_dim, seq_max_length).permute(0, 2, 1)
                sets = sets.reshape(num_batches, batch_size, seq_max_length, x_dim)

            else:
                # in this case, lb and hb will determine the low and high of the distribution for all each dimension
                sets_ = torch.zeros(num_batches * batch_size, seq_max_length, x_dim)
                for dim in range(x_dim):
                    # roll a fair dice to determine from which area to sample the entries for this dimension
                    # we need num_batches * batch_size samples (dice rolls) with possibilites being len(lb) options  of the dice roll
                    dice_roll = torch.randint(0, len(lbs), (num_batches * batch_size,))
                    for dice in range(len(lbs)):
                        sets_[dice_roll == dice, :, dim] = torch.rand((dice_roll == dice).sum(), seq_max_length) * (hb[dice][dim] - lb[dice][dim]) + lb[dice][dim]

                sets = sets_.reshape(num_batches, batch_size, seq_max_length, x_dim)

            return sets

        # random integer tensor of size num_batches x batch_size
        lengths = torch.randint(self.seq_min_length, self.seq_max_length + 1, (self.num_batches, self.batch_size))
        
        if self.architecture == "attention":
            hidden_tensor_seq_dims = (self.seq_max_length ** 2, self.phi_dim)
        elif self.architecture == "ssm" or self.architecture == "rnn":
            hidden_tensor_seq_dims = (self.seq_max_length, self.x_dim)
        elif self.architecture == "gpt2":
            hidden_tensor_seq_dims = (self.seq_max_length * self.architecture_config.config_decoder.n_layer, self.x_dim)
        else:
            hidden_tensor_seq_dims = (self.seq_max_length, self.phi_dim)

        outputs_tensor = torch.zeros(self.num_batches, self.batch_size, self.seq_max_length, self.y_dim)
        hidden_tensor = torch.zeros(self.num_batches, self.batch_size, hidden_tensor_seq_dims[0], hidden_tensor_seq_dims[1])

        if self.constraints is None or self.use_constraints is False:
            sets = torch.rand(self.num_batches, self.batch_size, self.seq_max_length, self.x_dim)
        else:
            sets = sample_with_constraints(self.constraints, lengths, self.num_batches, self.batch_size, self.seq_max_length, self.x_dim)
        seq_masks = torch.arange(self.seq_max_length).unsqueeze(0).unsqueeze(0) < lengths.unsqueeze(-1)

        indices = seq_masks.unsqueeze(-1).expand(-1, -1, -1, self.x_dim)
        sets[~indices] = 0
        if self.architecture != "gpt2":
            masks = seq_masks.unsqueeze(-1) * torch.tril(torch.ones(self.seq_max_length, self.seq_max_length)).unsqueeze(0).unsqueeze(0)
        else:
            masks = seq_masks

        # precomputing the data for the samples of the whole epoch
        for batch_id in tqdm(range(self.num_batches)):
            outputs_tensor[batch_id], hidden_tensor[batch_id] = self.mixing_fn(sets[batch_id], masks[batch_id]) # batch_size x seq_max_length x y_dim

        for batch_id in range(self.num_batches):
            yield {"input": sets[batch_id], "output": outputs_tensor[batch_id], "hidden": hidden_tensor[batch_id], "mask": masks[batch_id], "seq_mask": seq_masks[batch_id], "lengths": lengths[batch_id]}

    def __iter__(self):
        return self.generate_batch()


class SetDataModule(BaseStreamingPLDataModule):
    def __init__(self, dataset_parameters, **kwargs):
        super().__init__(**kwargs)
        self.dataset_parameters = dataset_parameters

        self.train_dataset = hydra.utils.instantiate(self.params['datasets']['train'], self.dataset_parameters, mixing_fn=None)
        self.val_dataset = hydra.utils.instantiate(self.params['datasets']['val'], self.dataset_parameters, mixing_fn=self.train_dataset.mixing_fn)
        self.test_dataset = hydra.utils.instantiate(self.params['datasets']['test'], self.dataset_parameters, mixing_fn=self.train_dataset.mixing_fn)
