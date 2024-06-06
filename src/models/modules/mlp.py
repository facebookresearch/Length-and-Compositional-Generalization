import torch
import torch.nn as nn
import hydra

class MLP(nn.Module):
    def __init__(self, x_dim, hid_dim, y_dim, n_hidden_layers, activation, **kwargs) -> None:
        super(MLP, self).__init__()

        self.x_dim = x_dim
        self.hid_dim = hid_dim
        self.y_dim = y_dim
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.sigmoid_activation = nn.Sigmoid()

        self.layers = nn.ModuleList()
        if self.n_hidden_layers == 0:
            self.layers.append(nn.Linear(x_dim, y_dim))
        else:
            self.layers.append(nn.Linear(x_dim, hid_dim))
            for _ in range(self.n_hidden_layers - 1):
                self.layers.append(nn.Linear(hid_dim, hid_dim))
            self.layers.append(nn.Linear(hid_dim, y_dim))
    
    def tempsigmoid(self, x):
        nd = 12.0 
        temp = nd / torch.log(torch.tensor(9.0)) 
        return torch.sigmoid(x/(temp))

    def forward(self, x_set):
        # x_set: [batch_size, max_seq_length, x_dim]
        for i, layer in enumerate(self.layers):
            # if i == len(self.layers) - 1:
            #     x_set = self.sigmoid_activation(layer(x_set))
            #     # x_set = self.tempsigmoid(layer(x_set))
            #     # x_set = layer(x_set)
            # else:
            #     x_set = self.activation(layer(x_set))
            x_set = self.activation(layer(x_set))
        return x_set

    def apply(self, fn):
        """
        Apply a function to all submodules of the Attention class.
        """
        for module in self.modules():
            fn(module)