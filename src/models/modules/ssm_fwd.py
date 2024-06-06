# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class SSMFwd(nn.Module):
    def __init__(self, x_dim, phi_dim, **kwargs) -> None:
        super(SSMFwd, self).__init__()

        self.x_dim = x_dim
        self.phi_dim = phi_dim

        self.A = nn.Parameter(torch.randn(self.x_dim, self.x_dim))
        self.B = nn.Parameter(torch.randn(self.x_dim, self.x_dim))
        self.Lambda = nn.Parameter(torch.randn(self.x_dim, self.x_dim))

        # make sure all the above matrices are orthogonal matrices
        self.A.data = torch.nn.init.orthogonal_(self.A.data)
        self.B.data = torch.nn.init.orthogonal_(self.B.data)
        self.Lambda.data = torch.nn.init.orthogonal_(self.Lambda.data)
        self.Lambda = torch.nn.init.orthogonal_(self.Lambda)

        # initialize the B and Lambda to be dense matrices of either +1, or -1
        # self.B.data = torch.randint(0, 2, (self.x_dim, self.x_dim)).float() * 2 - 1
        # self.Lambda.data = torch.randint(0, 2, (self.x_dim, self.x_dim)).float() * 2 - 1

    def forward(self, x_set):
        # x_set: [batch_size, max_seq_length, x_dim]

        hidden_states = []
        bs = x_set.shape[0]
        max_seq_length = x_set.shape[1]

        # import time
        # start = time.perf_counter()
        for T in range(max_seq_length):
            raw_hidden_states = []
            for j in range(T+1):
                # raw_hidden_states.append(torch.matmul(torch.matmul(x_set[:, T-j], self.B), torch.pow(self.Lambda, j)))
                raw_hidden_states.append(torch.bmm(torch.matrix_power(self.Lambda, j).unsqueeze(0).repeat(bs, 1, 1), torch.bmm(self.B.unsqueeze(0).repeat(bs, 1, 1), x_set[:, T-j].unsqueeze(-1))).squeeze())

            # stack and sum along the T dimension
            raw_hidden_states = torch.stack(raw_hidden_states).sum(dim=0) # [batch_size, x_dim]
            hidden_states.append(raw_hidden_states)
        hidden_states = torch.stack(hidden_states).permute(1, 0, 2) # [batch_size, max_seq_length, x_dim]
        # end = time.perf_counter()

        # without for loops TODO: strangely slower, about 1.5~2 times.
        # create a lower traingular matrix of size T x T x 4 x 4 such that the diagonal and the lower triangle are filled with the powers of Lambda
        # and the upper triangle is filled with zeros
        # it should be as follows:
        # [Lambda^0 0-matrix 0-matrix 0-matrix]
        # [Lambda^1 Lambda^0 0-matrix 0-matrix]
        # [Lambda^2 Lambda^1 Lambda^0 0-matrix]
        # ...
        # [Lambda^9 Lambda^8 Lambda^7 Lambda^6 Lambda^5 Lambda^4 Lambda^3 Lambda^2 Lambda^1 Lambda^0]

        # Lambda_powers_tensor_list = [torch.stack([self.Lambda**(j-i) for i in range(j+1)] + [torch.zeros(self.x_dim, self.x_dim)] * (max_seq_length - 1 - j)) for j in range(max_seq_length)]
        # Lambda_powers_lower_triangular = torch.stack(Lambda_powers_tensor_list) # T x T x x_dim x x_dim
        # Bx = torch.einsum('ij,klj->kli', self.B, x_set) # [batch_size, x_dim, max_seq_length]
        # hidden_states_ = torch.einsum('ijklm,ijkmq->ijklq', Lambda_powers_lower_triangular.unsqueeze(0).repeat(bs, 1, 1, 1, 1), Bx.unsqueeze(1).repeat(1, x_set.shape[1], 1, 1).unsqueeze(-1)).squeeze().sum(dim=-2)
        # # end_ = time.perf_counter()
        # # print(f"Elapsed time without for loops: {end_-end}")
        # # print(f"Elapsed time with for loops: {end-start}")
        # # print(torch.allclose(hidden_states, hidden_states_, atol=1e-5))
        # hidden_states = hidden_states_.clone()

        return hidden_states # [batch_size, max_seq_length, x_dim]


    def apply(self, fn):
        """
        Apply a function to all submodules of the Attention class.
        """
        for module in [self.A, self.B, self.Lambda]:
            fn(module)
