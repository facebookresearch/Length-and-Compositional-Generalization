import torch
import torch.nn as nn
import hydra

class RNNFwd(nn.Module):
    def __init__(self, x_dim, phi_dim, activation, **kwargs) -> None:
        super(RNNFwd, self).__init__()

        self.x_dim = x_dim
        self.phi_dim = phi_dim
        self.activation = activation

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

        for T in range(max_seq_length):
            hidden_states.append(self.activation(torch.bmm(self.B.unsqueeze(0).repeat(bs, 1, 1), x_set[:, T].unsqueeze(-1)) + torch.bmm(self.Lambda.unsqueeze(0).repeat(bs, 1, 1), hidden_states[-1] if T > 0 else torch.zeros(bs, self.x_dim, 1).to(x_set.device)))) # [batch_size, x_dim, 1]
        
        hidden_states = torch.stack(hidden_states).squeeze(-1).permute(1, 0, 2) # [batch_size, max_seq_length, x_dim]

        return hidden_states # [batch_size, max_seq_length, x_dim]


    def apply(self, fn):
        """
        Apply a function to all submodules of the Attention class.
        """
        for module in [self.A, self.B, self.Lambda]:
            fn(module)
