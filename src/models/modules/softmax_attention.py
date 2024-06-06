# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from src.models.modules.attention import Attention

class SoftmaxAttention(Attention):
    def __init__(self, x_dim, phi_dim, **kwargs) -> None:
        super(SoftmaxAttention, self).__init__(x_dim=x_dim, phi_dim=phi_dim, **kwargs)

        self.attention_fn_key = "softmax"
        self.attention_fn = nn.Softmax(dim=-1)

    def forward(self, x_set):
        batch_size, max_seq_len, x_dim = x_set.size()

        query = self.query_projection(x_set) # [batch_size, max_seq_len, phi_dim]
        key = self.key_projection(x_set) # [batch_size, max_seq_len, phi_dim]
        value = self.value_projection(x_set) # [batch_size, max_seq_len, phi_dim]

        # compute all causal attention scores
        # mask the non-causal attention scores with -inf and apply softmax
        dot_products = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.phi_dim).float()) # [batch_size, max_seq_len, max_seq_len]
        attention_scores = self.attention_fn(dot_products.masked_fill(torch.triu(torch.ones(max_seq_len, max_seq_len).to(x_set.device), diagonal=1).bool(), -float('inf'))) # [batch_size, max_seq_len, max_seq_len]

        attended_values = torch.zeros(batch_size, max_seq_len, self.phi_dim).to(x_set.device) # [batch_size, max_seq_len, phi_dim]

        # compute the weighted sum of the values given the attention scores
        # i.e., attended_values[:, i, :] = \sum_{j=0}^{max_seq_len} attention_scores[:, i, j] * value[:, j, :]
        for i in range(max_seq_len):
            attended_values[:, i, :] = torch.bmm(attention_scores[:, i, :].unsqueeze(1), value).squeeze(1)

        return attended_values # [batch_size, max_seq_len, phi_dim]


class MultiLayerSoftmaxAttention(nn.Module):
    def __init__(self, x_dim, phi_dim, num_layers, **kwargs) -> None:
        super(MultiLayerSoftmaxAttention, self).__init__()

        self.attentions = nn.ModuleList([SoftmaxAttention(x_dim=x_dim, phi_dim=phi_dim, **kwargs) for _ in range(num_layers)])

    def forward(self, x_set):
        for attention in self.attentions:
            x_set = attention(x_set) + x_set # skip connection
        return x_set # [batch_size, max_seq_len, phi_dim]