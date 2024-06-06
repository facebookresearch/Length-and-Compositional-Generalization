# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from typing import Optional, Callable
import torch.nn as nn
import hydra
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from src.utils.general import get_pylogger

log = get_pylogger(__name__)

class BaseStreamingDataset(IterableDataset):
    """
    A streaming dataset implements the __iter__ method, which returns an iterator
    that yields samples from the dataset.
    """

    def __init__(self, dataset_parameters, **kwargs):
        super().__init__()
        self.dataset_parameters = dataset_parameters
        self.params = kwargs
        self.random_state = np.random.RandomState(self.dataset_parameters["seed"])


    def generate_batch(self):
        raise NotImplementedError()

    def __iter__(self):
        # This method should return an iterator that yields samples from the dataset.
        # The exact implementation will depend on how your data is stored.
        return self.generate_batch()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=self.init_mean, std=self.init_std)
            torch.nn.init.constant_(m.bias, 0.0)
            m.weight.requires_grad = False
            m.bias.requires_grad = False
        elif type(m) == nn.Parameter:
            torch.nn.init.orthogonal_(m)
            m.requires_grad = False

    def mixing(self, mixing_architecture_config):

        self.mixing_type = mixing_architecture_config["name"]
        if mixing_architecture_config["name"] != "gpt2":
            self.init_mean = mixing_architecture_config["init_mean"]
            self.init_std = mixing_architecture_config["init_std"]
            self.load = mixing_architecture_config["load"]

            self.phi_individual = mixing_architecture_config["phi_individual"]
            self.phi_aggregate = mixing_architecture_config["phi_aggregate"]

            if self.load == False:
                # initialize the weights
                self.phi_individual.apply(self.init_weights)
                self.phi_aggregate.apply(self.init_weights)

                # save the mixing function modules
                torch.save(self.phi_individual, "phi_individual.pt")
                torch.save(self.phi_aggregate, "phi_aggregate.pt")
            
            else:
                # the path for the modules is at mixing_architecture_config["phi_individual_path"] and mixing_architecture_config["phi_aggregate_path"]
                self.phi_individual = torch.load(mixing_architecture_config["phi_individual_path"])
                self.phi_aggregate = torch.load(mixing_architecture_config["phi_aggregate_path"])


        if self.mixing_type == "deepset":
            return self.deepset_mixing
        elif self.mixing_type == "attention":
            return self.attention_mixing
        elif self.mixing_type == "softmax_attention":
            return self.softmax_attention_mixing
        elif self.mixing_type == "ssm":
            return self.ssm_mixing
        elif self.mixing_type == "rnn":
            return self.rnn_mixing
        elif self.mixing_type == "gpt2":
            from transformers import GPT2Model, GPT2Config
            self.gpt2 = GPT2Model(mixing_architecture_config["config_decoder"])            
            return self.gpt2_mixing


    def deepset_mixing(self, set_, mask):
        # set_ : batch_size x max_seq_length x x_dim
        # mask : batch_size x max_seq_length x max_seq_length

        phi_individual_output = self.phi_individual(set_) # batch_size x max_seq_length x phi_dim
        phi_individual_masked = phi_individual_output.unsqueeze(2).repeat(1, 1, mask.shape[2], 1) * mask.unsqueeze(-1).repeat(1, 1, 1, self.phi_dim) # batch_size x max_seq_length x max_seq_length x phi_dim
        # phi_aggregate_output = torch.sigmoid(self.phi_aggregate(torch.sum(phi_individual_masked, dim=2))) # batch_size x max_seq_length x phi_dim
        phi_aggregate_output = self.phi_aggregate(torch.sum(phi_individual_masked, dim=2)) # batch_size x max_seq_length x phi_dim
        non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float() # batch_size x max_seq_length x 1
        return phi_aggregate_output * non_zero_mask, phi_individual_output # batch_size x max_seq_length x phi_dim

    def attention_mixing(self, set_, mask):
        # set_ : batch_size x max_seq_length x x_dim
        # mask : batch_size x max_seq_length x max_seq_length

        # phi_individual's output is attended_values of shape (batch_size, max_seq_length, max_seq_length, phi_dim), i.e., phi_individual_output[:, i, j, :] is the attended value when considering the attention of token i on j
        phi_individual_output = self.phi_individual(set_)  # [batch_size, max_seq_len, max_seq_len, phi_dim]
        phi_individual_masked = phi_individual_output * mask.unsqueeze(-1).repeat(1, 1, 1, self.phi_dim)  # [batch_size, max_seq_len, max_seq_len, phi_dim]
        phi_aggregate_output = self.phi_aggregate(torch.sum(phi_individual_masked, dim=2))  # [batch_size, max_seq_len, phi_dim]
        non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float()  # [batch_size, max_seq_len, 1]
        return phi_aggregate_output * non_zero_mask, phi_individual_output.reshape(set_.shape[0], -1, phi_individual_output.shape[-1]) # phi_aggregate_output: [batch_size, max_seq_len, phi_dim]

    def softmax_attention_mixing(self, set_, mask):
        # set_ : batch_size x max_seq_length x x_dim
        # mask : batch_size x max_seq_length x max_seq_length

        # phi_individual's output is attended_values of shape (batch_size, max_seq_length, phi_dim)
        # there is no summation here, the weights are already applied in the phi_individual (softmax)

        phi_individual_output = self.phi_individual(set_) # [batch_size, max_seq_len, phi_dim]
        # phi_aggregate_output = torch.sigmoid(self.phi_aggregate(phi_individual_output))  # [batch_size, max_seq_len, phi_dim]
        phi_aggregate_output = self.phi_aggregate(phi_individual_output)  # [batch_size, max_seq_len, phi_dim]
        non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float()  # [batch_size, max_seq_len, 1]
        return phi_aggregate_output * non_zero_mask, phi_individual_output # [batch_size, max_seq_len, phi_dim]


    def ssm_mixing(self, set_, mask):
        # set_ : batch_size x max_seq_length x x_dim
        # mask : batch_size x max_seq_length x max_seq_length

        phi_individual_output = self.phi_individual(set_)  # [batch_size, max_seq_length, x_dim] contains \sum_{0}^{T-1}(\Lambda^j * B * x_{T-j})
        # because the summation is taken care of in the phi_individual, we don't need to do any summation here
        phi_aggregate_output = self.phi_aggregate(phi_individual_output)  # [batch_size, max_seq_length, phi_dim]
        non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float()  # [batch_size, max_seq_length, 1]
        return phi_aggregate_output * non_zero_mask, phi_individual_output # phi_aggregate_output: [batch_size, max_seq_length, phi_dim]
        
    def rnn_mixing(self, set_, mask):
        # set_ : batch_size x max_seq_length x x_dim
        # mask : batch_size x max_seq_length x max_seq_length

        phi_individual_output = self.phi_individual(set_) # [batch_size, max_seq_length, x_dim] contains h1, h2, ..., hT
        phi_aggregate_output = self.phi_aggregate(phi_individual_output)  # [batch_size, max_seq_length, phi_dim]
        non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float()  # [batch_size, max_seq_length, 1]
        return phi_aggregate_output * non_zero_mask, phi_individual_output # phi_aggregate_output: [batch_size, max_seq_length, phi_dim]

    def gpt2_mixing(self, set_, mask):
        # set_ : batch_size x max_seq_length x x_dim
        # mask : batch_size x max_seq_length

        with torch.no_grad():
            outputs = self.gpt2(inputs_embeds=set_, attention_mask=mask, output_hidden_states=True) # ['last_hidden_state', 'past_key_values', 'hidden_states']
            # outputs["last_hidden_state"]: [batch_size, max_seq_length, x_dim]
            # outputs["hidden_states"]: tuple of num_layers each [batch_size, max_seq_length, x_dim]
            # outputs["past_key_values"]: tuple of num_layers each a tuple of size 2 each [batch_size, num_heads, max_seq_length, head_dim] (where num_heads x head_dim = x_dim)
            hidden_states = outputs["hidden_states"][1:] # num_layers * [batch_size, max_seq_length, x_dim]
            # concat hidden states from the tuple above along dim=1
            hidden_states = torch.cat(hidden_states, dim=1) # [batch_size, num_layers * max_seq_length, x_dim]
            non_zero_mask = mask.float().unsqueeze(-1).repeat(1, 1, outputs["last_hidden_state"].shape[-1])  # [batch_size, max_seq_length, x_dim]
            return outputs["last_hidden_state"] * non_zero_mask, hidden_states # [batch_size, max_seq_length, x_dim]


class BaseStreamingPLDataModule(LightningDataModule, ABC):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(self, seed: int, dataset_parameters: dict = None, num_workers: int = 0, **kwargs):
        """

        Parameters
        ----------
        seed : Random seed
        num_workers : Setting num_workers as a positive integer will turn on multiprocess data loading with the specified number of loader worker processes

        kwargs: dataset specific parameters

        Returns
        -------
        An instance of the dataset that extends pytorch_lightning.DataModule
        """
        super().__init__()
        self.dataset_parameters = dataset_parameters
        self.datasets = kwargs["datasets"]
        self.params = kwargs

        self.seed = seed

        self.train_dataset: Optional[IterableDataset] = None
        self.val_dataset: Optional[IterableDataset] = None
        self.test_dataset: Optional[IterableDataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        
        assert stage in set(["fit", "validate", "test", None])

        if self.train_dataset is None:
            self.train_dataset = hydra.utils.instantiate(
                self.datasets["train"])
            log.info("The train dataset has been instantiated.")
            
        if self.val_dataset is None:
            self.val_dataset = hydra.utils.instantiate(
                self.datasets["val"])
            log.info("The validation dataset has been instantiated.")

        if (stage == "test" or stage is None) and self.test_dataset is None:
            self.test_dataset = hydra.utils.instantiate(
                self.datasets["test"])
            log.info("The test dataset has been instantiated.")

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        # don't use shuffle and batch size with streaming datasets when using dataloader
        return DataLoader(
            dataset=self.train_dataset,
            num_workers=self.dataset_parameters["num_workers"],
            drop_last=False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            num_workers=self.dataset_parameters["num_workers"],
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            num_workers=self.dataset_parameters["num_workers"],
            drop_last=False,
        )
