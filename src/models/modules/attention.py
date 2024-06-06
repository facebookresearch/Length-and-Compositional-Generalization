# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

class Pass(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(Pass, self).__init__()

    def forward(self, x_set):
        return x_set

    def apply(self, fn):
        """
        Apply a function to all submodules of the Attention class.
        """
        for module in self.modules():
            fn(module)

class Attention(nn.Module):
    def __init__(self, x_dim, phi_dim, **kwargs) -> None:
        super(Attention, self).__init__()

        # Linear projections for queries, keys, and values
        self.query_projection = nn.Linear(x_dim, phi_dim)
        self.key_projection = nn.Linear(x_dim, phi_dim)
        self.value_projection = nn.Linear(x_dim, phi_dim)
        self.phi_dim = phi_dim
        self.x_dim = x_dim

        self.attention_fn_key = kwargs.get("attention_fn", "sigmoid")
        if self.attention_fn_key == "sigmoid":
            self.attention_fn = nn.Sigmoid()
        elif self.attention_fn_key == "relu":
            self.attention_fn = nn.ReLU()
        elif self.attention_fn_key == "linear":
            self.attention_fn = Pass()
            # self.attention_fn = lambda x: x
        else:
            raise ValueError(f"Attention function {self.attention_fn_key} not supported.")

    def forward(self, x_set):
        batch_size, max_seq_len, x_dim = x_set.size()

        query = self.query_projection(x_set) # [batch_size, max_seq_len, phi_dim]
        key = self.key_projection(x_set) # [batch_size, max_seq_len, phi_dim]
        value = self.value_projection(x_set) # [batch_size, max_seq_len, phi_dim]

        # compute all causal attention scores
        attention_scores = self.attention_fn(torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.phi_dim).float())) # [batch_size, max_seq_len, max_seq_len]
        # Iterate through pairs of vectors
        attended_values_ = torch.zeros(batch_size, max_seq_len, max_seq_len, self.phi_dim).to(x_set.device) # [batch_size, max_seq_len, max_seq_len, phi_dim]
        for i in range(max_seq_len):
            for j in range(i+1):
                # fill in the attended values with the attention score of i on j times the value at j
                # i.e., attended_values_[:, i, j, :] = attention_scores[:, i, j] * value[:, j, :]
                # attention_scores[:, i, j].unsqueeze(1): [batch_size, 1]
                # value[:, j, :]: [batch_size, phi_dim]
                attended_values_[:, i, j, :] = attention_scores[:, i, j].unsqueeze(1) * value[:, j, :] # [batch_size, 1] * [batch_size, phi_dim] --> [batch_size, phi_dim]

        return attended_values_


    def apply(self, fn):
        """
        Apply a function to all submodules of the Attention class.
        """
        for module in self.modules():
            fn(module)
