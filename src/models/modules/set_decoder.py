from omegaconf import OmegaConf
import hydra
import torch
import torch.nn as nn
from torch.nn import MSELoss

class SetDecoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        decoder_config = kwargs.get("decoder_config")
        phi_individual_config = decoder_config.get("phi_individual")
        phi_aggregate_config = decoder_config.get("phi_aggregate")

        self.name = decoder_config.get("name")
        self.phi_dim = kwargs.get("phi_dim")
        self.x_dim = kwargs.get("x_dim")
        self.y_dim = kwargs.get("y_dim")
        self.init_mean = decoder_config.get("init_mean", 0.0)
        self.init_std = decoder_config.get("init_std", 0.1)
        self.loss = MSELoss()
        self.n_layers = decoder_config.get("n_layers", 1)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            # each layer should have its own phi_individual and phi_aggregate
            self.layers.append(
                SetDecoderLayer(
                    name=self.name,
                    phi_individual_config=phi_individual_config,
                    phi_aggregate_config=phi_aggregate_config,
                    d_in=self.x_dim if i == 0 else self.phi_dim,
                    d_hid=self.phi_dim,
                    d_out=self.phi_dim if i != self.n_layers - 1 else self.y_dim,
                    init_mean=self.init_mean,
                    init_std=self.init_std,
                )
            )


    def forward(self, x_set, mask, y_set=None):

        # return self.phi_aggregate(self.phi_individual(x_set, mask), mask)
        y_preds, hidden = self._forward(x_set, mask)
        losses = []
        for i in range(y_preds.shape[1]):
            losses.append(self.loss(y_preds[:, i], y_set[:, i]))
        losses = torch.stack(losses)
        loss = losses.mean()

        return {"y_pred": y_preds, "hidden": hidden, "loss": loss}


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=self.init_mean, std=self.init_std)
            torch.nn.init.constant_(m.bias, 0.0)
            m.weight.requires_grad = True
            m.bias.requires_grad = True
        elif type(m) == nn.Parameter:
            torch.nn.init.orthogonal_(m)
            m.requires_grad = True


class SetDecoderLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.name = kwargs.get("name")
        self.phi_dim = kwargs.get("phi_dim")
        self.d_in = kwargs.get("d_in")
        self.d_hid = kwargs.get("d_hid")
        self.d_out = kwargs.get("d_out")
        # kwargs.get("phi_individual_config")["Linear1"]["in_features"] = self.d_in
        # kwargs.get("phi_aggregate_config")["Linear1"]["out_features"] = self.d_out
        self.init_mean = kwargs.get("init_mean")
        self.init_std = kwargs.get("init_std")

        self.phi_individual = hydra.utils.instantiate(kwargs.get("phi_individual_config"))
        self.phi_aggregate = hydra.utils.instantiate(kwargs.get("phi_aggregate_config"))
        # if self.name == "deepset":
        #     self.phi_individual = torch.nn.Sequential(
        #         # *[hydra.utils.instantiate(layer_config) for _, layer_config in phi_individual_config.items()]
        #         *[hydra.utils.instantiate(layer_config) for _, layer_config in kwargs.get("phi_individual_config").items()]
        #     )
        # else:
        #     # self.phi_individual = hydra.utils.instantiate(phi_individual_config)
        #     self.phi_individual = hydra.utils.instantiate(kwargs.get("phi_individual_config"))
            
        # self.phi_aggregate = torch.nn.Sequential(
        #     # *[hydra.utils.instantiate(layer_config) for _, layer_config in phi_aggregate_config.items()]
        #     *[hydra.utils.instantiate(layer_config) for _, layer_config in kwargs.get("phi_aggregate_config").items()]
        # )

        # initialize the weights
        self.phi_individual.apply(self.init_weights)
        self.phi_aggregate.apply(self.init_weights)


    def forward(self, x_set, mask):

        return self.phi_aggregate(self.phi_individual(x_set, mask), mask)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=self.init_mean, std=self.init_std)
            torch.nn.init.constant_(m.bias, 0.0)
            m.weight.requires_grad = True
            m.bias.requires_grad = True
        elif type(m) == nn.Parameter:
            torch.nn.init.orthogonal_(m)
            m.requires_grad = True



class DeepSet(SetDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.A = nn.Parameter(torch.randn(kwargs["x_dim"], kwargs["x_dim"]))

    def _forward(self, x_set, mask):
    
        for layer in self.layers:
            phi_individual_output = layer.phi_individual(x_set) # batch_size x max_seq_length x phi_dim
            phi_individual_masked = phi_individual_output.unsqueeze(2).repeat(1, 1, mask.shape[2], 1) * mask.unsqueeze(-1).repeat(1, 1, 1, self.phi_dim) # batch_size x max_seq_length x max_seq_length x phi_dim
            # phi_aggregate_output = torch.sigmoid(layer.phi_aggregate(torch.sum(phi_individual_masked, dim=2))) # batch_size x max_seq_length x phi_dim
            phi_aggregate_output = layer.phi_aggregate(torch.sum(phi_individual_masked, dim=2)) # batch_size x max_seq_length x phi_dim
            non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float() # batch_size x max_seq_length x 1
            y_preds = phi_aggregate_output * non_zero_mask # batch_size x max_seq_length x phi_dim
            x_set = y_preds.clone()

        return y_preds, phi_individual_output

    # def forward(self, x_set, mask, y_set=None):

        # x_set: [batch_size, max_seq_length, x_dim]
        # mask: [batch_size, max_seq_length, max_seq_length]
        # y_set: [batch_size, max_seq_length, y_dim]
        
        # y_preds = []    
        # for i in range(mask.shape[1]):
        #     y_preds.append(
        #         self.phi_aggregate(
        #         torch.sum(
        #             self.phi_individual(x_set) * mask[:, i].unsqueeze(-1).repeat(1, 1, self.phi_dim),
        #             dim=1,
        #         )
        #     ) * (mask[:, i].sum(dim=1, keepdim=True) > 0).float()
        #     )
        # y_preds = torch.stack(y_preds).permute(1, 0, 2) # batch_size x seq_max_length-1 x y_dim


class AttentionSet(SetDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # in this case, phi_aggregate is a linear layer, and phi_individual is a multi-head
        # attention layer that acts on pairs of elements in the set

    def _forward(self, x_set, mask):

        for layer in self.layers:

            # phi_individual_output[:, i, j, :] is the attended value when considering the attention of token i on j
            phi_individual_output = layer.phi_individual(x_set) # [batch_size, max_seq_len, max_seq_len, phi_dim]
            phi_individual_masked = phi_individual_output * mask.unsqueeze(-1).repeat(1, 1, 1, self.phi_dim)  # [batch_size, max_seq_len, max_seq_len, phi_dim]
            phi_aggregate_output = layer.phi_aggregate(torch.sum(phi_individual_masked, dim=2))  # [batch_size, max_seq_len, phi_dim]
            non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float()  # [batch_size, max_seq_len, 1]
            y_preds = phi_aggregate_output * non_zero_mask # [batch_size, max_seq_len, phi_dim]
            x_set = y_preds.clone()

        return y_preds, phi_individual_output.reshape(x_set.shape[0], -1, phi_individual_output.shape[-1])

    # def forward(self, x_set, mask, y_set=None):

        # x_set: [batch_size, max_seq_length, x_dim]
        # mask: [batch_size, max_seq_length, max_seq_length]
        # y_set: [batch_size, max_seq_length, y_dim]

        # y_preds = []
        # phi_individual_output = self.phi_individual(x_set) # [batch_size, max_seq_len, max_seq_len, phi_dim]
        # for i in range(mask.shape[1]):
        #     y_preds.append(
        #         self.phi_aggregate(
        #             torch.sum(
        #                 phi_individual_output[:, i] * mask[:, i].unsqueeze(-1).repeat(1, 1, self.phi_dim),
        #                 dim=1,
        #         )
        #         ) * (mask[:, i].sum(dim=1, keepdim=True) > 0).float()
        #     )
        # y_preds = torch.stack(y_preds).permute(1, 0, 2) # batch_size x seq_max_length-1 x y_dim

        # return {"y_pred": y_preds, "loss": loss}


class SoftmaxAttentionSet(SetDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # in this case, phi_aggregate is a linear layer, and phi_individual is a multi-head
        # attention layer that acts on pairs of elements in the set

    def _forward(self, x_set, mask):

        skip_connections = torch.zeros(x_set.shape).to(x_set.device)

        for layer in self.layers:
            phi_individual_output = layer.phi_individual(x_set + skip_connections)
            phi_aggregate_output = layer.phi_aggregate(phi_individual_output)  # [batch_size, max_seq_len, phi_dim]
            non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float()  # [batch_size, max_seq_len, 1]
            y_preds = phi_aggregate_output * non_zero_mask # [batch_size, max_seq_len, x_dim]
            skip_connections = y_preds.clone()
            x_set = y_preds.clone()

        return y_preds, phi_individual_output


    # def forward(self, x_set, mask, y_set=None):

        # x_set: [batch_size, max_seq_length, x_dim]
        # mask: [batch_size, max_seq_length, max_seq_length]
        # y_set: [batch_size, max_seq_length, y_dim]

        # y_preds = []
        # phi_individual_output = self.phi_individual(x_set) # [batch_size, max_seq_len, phi_dim]
        # for i in range(mask.shape[1]):
        #     y_preds.append(
        #         self.phi_aggregate(
        #                 phi_individual_output[:, i]
        #         ) * (mask[:, i].sum(dim=1, keepdim=True) > 0).float()
        #     )
        # y_preds = torch.stack(y_preds).permute(1, 0, 2) # batch_size x seq_max_length x y_dim

        # y_preds = self._forward(x_set, mask)
    
        # losses = []
        # for i in range(y_preds.shape[1]):
        #     losses.append(self.loss(y_preds[:, i], y_set[:, i]))
        # losses = torch.stack(losses)
        # loss = losses.mean()

        # return {"y_pred": y_preds, "loss": loss}


class SSMSet(SetDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _forward(self, x_set, mask):

        for layer in self.layers:
            phi_individual_output = layer.phi_individual(x_set) # [batch_size, max_seq_length, x_dim] contains \sum_{0}^{T-1}(\Lambda^j * B * x_{T-j})
            phi_aggregate_output = layer.phi_aggregate(phi_individual_output)
            non_zero_mask = (mask.sum(dim=2, keepdim=True) > 0).float()
            y_preds = phi_aggregate_output * non_zero_mask  # [batch_size, max_seq_length, phi_dim]
            x_set = y_preds.clone()

        return y_preds, phi_individual_output

    # def forward(self, x_set, mask, y_set=None):

        # x_set: [batch_size, max_seq_length, x_dim]
        # mask: [batch_size, max_seq_length, max_seq_length]
        # y_set: [batch_size, max_seq_length, y_dim]

        # y_preds = []
        # phi_individual_output = self.phi_individual(x_set) # [batch_size, max_seq_len, phi_dim] contains \sum_{0}^{T-1}(\Lambda^j * B * x_{T-j})
        # for i in range(mask.shape[1]):
        #     y_preds.append(
        #         # self.phi_aggregate(
        #         #     torch.sum(
        #         #         phi_individual_output * mask[:, i].unsqueeze(-1).repeat(1, 1, self.phi_dim),
        #         #         dim=1,
        #         # )
        #         self.phi_aggregate(phi_individual_output[:, i]) * (mask[:, i].sum(dim=1, keepdim=True) > 0).float()
        #     )
        # y_preds = torch.stack(y_preds).permute(1, 0, 2) # batch_size x seq_max_length x y_dim

        # return {"y_pred": y_preds, "loss": loss}


class RNNSet(SetDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _forward(self, x_set, mask):

        for layer in self.layers:
            phi_individual_output = layer.phi_individual(x_set) # [batch_size, max_seq_length, x_dim] contains h1, h2, h3, ...
            y_preds = layer.phi_aggregate(phi_individual_output) * (mask.sum(dim=2, keepdim=True) > 0).float() # [batch_size, max_seq_length, phi_dim]
            x_set = y_preds.clone()

        return y_preds, phi_individual_output

    # def forward(self, x_set, mask, y_set=None):

        # x_set: [batch_size, max_seq_length, x_dim]
        # mask: [batch_size, max_seq_length, max_seq_length]
        # y_set: [batch_size, max_seq_length, y_dim]

        # return {"y_pred": y_preds, "loss": loss}


class GPT2(nn.Module):

    def __init__(self, **kwargs) -> None:
        super().__init__()

        decoder_config = kwargs.get("decoder_config")
        self.name = decoder_config.get("name")
        self.phi_dim = kwargs.get("phi_dim")
        self.x_dim = kwargs.get("x_dim")
        self.y_dim = kwargs.get("y_dim")
        self.loss = MSELoss()

        from transformers import GPT2Model, GPT2Config
        config = hydra.utils.instantiate(decoder_config["config_decoder"])
        self.decoder = GPT2Model(config)


    def _forward(self, x_set, mask):

        outputs = self.decoder(inputs_embeds=x_set, attention_mask=mask, output_hidden_states=True) # ['last_hidden_state', 'past_key_values', 'hidden_states']
        # outputs["last_hidden_state"]: [batch_size, max_seq_length, x_dim]
        # outputs["hidden_states"]: tuple of num_layers each [batch_size, max_seq_length, x_dim]
        # outputs["past_key_values"]: tuple of num_layers each a tuple of size 2 each [batch_size, num_heads, max_seq_length, head_dim] (where num_heads x head_dim = x_dim)
        hidden_states = outputs["hidden_states"][1:] # num_layers * [batch_size, max_seq_length, x_dim]
        # concat hidden states from the tuple above along dim=1
        hidden_states = torch.cat(hidden_states, dim=1) # [batch_size, num_layers * max_seq_length, x_dim]
        non_zero_mask = mask.float().unsqueeze(-1).repeat(1, 1, outputs["last_hidden_state"].shape[-1])  # [batch_size, max_seq_length, x_dim]
        return outputs["last_hidden_state"] * non_zero_mask, hidden_states # [batch_size, max_seq_length, x_dim]


    def forward(self, x_set, mask, y_set=None):

        y_preds, hidden = self._forward(x_set, mask)
        losses = []
        for i in range(y_preds.shape[1]):
            losses.append(self.loss(y_preds[:, i], y_set[:, i]))
        losses = torch.stack(losses)
        loss = losses.mean()

        return {"y_pred": y_preds, "hidden": hidden, "loss": loss}