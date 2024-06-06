# Provable Length and Compositional Generalization

To reproduce, please run the following commands after installing and activating the environments with the `requirements.txt`


### Length Generalization

The following command trains a Deep set model with 1 hidden layer MLPs for $\omega,\psi$ to predict the output sequences of another Deep set in the realizable setting, i.e., the MLPs for $\rho,\phi$ will also have 1 hidden layer. In the following, input, output, and hidden dimension are all the same $n=m=k=20$, and training sequences are up to length $T=10$, and evaluation is done on sequences of length $T=20$.

```
python3 run_train.py +experiment=deepset.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.mixing_architecture.phi_individual.n_hidden_layers=1 datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_individual.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=2 datamodule.datasets.train.seq_max_length=10 datamodule.datasets.val.seq_min_length=20 datamodule.datasets.val.seq_max_length=20 datamodule.datasets.test.seq_min_length=20 datamodule.datasets.test.seq_max_length=20 trainer.min_epochs=100`
```

For evaluating on other lengths, you can use the saved checkpoint for the above training, (as well as its datamodule with $\rho,\phi$ automatically loaded from the checkpoint path), and run the following commands to do inference with different sequence lengths.

```
CKPT="path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=deepset datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=256 "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=100
```

For Transformer with softmax attention, SSM, and RNN, you can use similar commands with small modifications to adapt the datamodule and the learner $h$.

#### Transformer
```
# Training

python3 run_train.py +experiment=softmax_attention.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=2 datamodule.datasets.train.seq_max_length=10 datamodule.datasets.val.seq_min_length=20 datamodule.datasets.val.seq_max_length=20 datamodule.datasets.test.seq_min_length=20 datamodule.datasets.test.seq_max_length=20

# Evaluation

CKPT="path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=softmax_attention datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=256 "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=100
```

#### SSM
```
# Training

python3 run_train.py +experiment=ssm.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=2 datamodule.datasets.train.seq_max_length=10 datamodule.datasets.val.seq_min_length=20 datamodule.datasets.val.seq_max_length=20 datamodule.datasets.test.seq_min_length=20 datamodule.datasets.test.seq_max_length=20

# Evaluation

CKPT="path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=ssm datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=256 "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=100
```

#### RNN
```
# Training

python3 run_train.py +experiment=rnn.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=2 datamodule.datasets.train.seq_max_length=10 datamodule.datasets.val.seq_min_length=20 datamodule.datasets.val.seq_max_length=20 datamodule.datasets.test.seq_min_length=20 datamodule.datasets.test.seq_max_length=20

# Evaluation

CKPT="path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=rnn datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=256 "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=100
```
---------
### Compositional Generalization

Running the following command with `datamodule.datasets.train.use_constraints=True` will use the constraints for sampling training and testing sequences for constructing the datasets, and will train a Deep set model with 1 hidden layer MLPs for $\omega,\psi$ to predict the output sequences of another Deep set in the realizable setting, i.e., the MLPs for $\rho,\phi$ will also have 1 hidden layer. Again, the input, output, and hidden dimension are all the same $n=m=k=20$, and training sequences are up to length $T=10$, and evaluation is done on sequences of the same length $T=10$. You could however modify `datamodule.datasets.test.seq_min_length=10` to any longer length, and observe that compositional generalization at longer lengths can also be achieved.

```
python3 run_train.py +experiment=deepset.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.datasets.train.use_constraints=True datamodule.mixing_architecture.phi_individual.n_hidden_layers=1 datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_individual.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=10 datamodule.datasets.train.seq_max_length=10
```

Alternatively, for doing inference on other lengths using the saved checkpoint (as well as its datamodule with $\rho,\phi$ automatically loaded from the checkpoint path), you may run the following commands.

```
CKPT="/path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=deepset datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=$BSIZE datamodule.dataset_parameters.num_workers=$NUM_WORKERS "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=10
```

For Transformer with softmax attention, SSM, and RNN, you can use similar commands with small modifications to adapt the datamodule and the learner $h$.

#### Transformer
```
# Training

python3 run_train.py +experiment=softmax_attention.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.datasets.train.use_constraints=True datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=10 datamodule.datasets.train.seq_max_length=10

# Evaluation

CKPT="path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=softmax_attention datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=256 "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=100
```

#### SSM
```
# Training

python3 run_train.py +experiment=ssm.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.datasets.train.use_constraints=True datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=10 datamodule.datasets.train.seq_max_length=10

# Evaluation

CKPT="path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=ssm datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=256 "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=100
```

#### RNN
```
# Training

python3 run_train.py +experiment=rnn.yaml trainer.accelerator="cpu" trainer.devices="auto" datamodule.datasets.train.use_constraints=True datamodule.mixing_architecture.phi_aggregate.n_hidden_layers=1 model.set_decoder.decoder_config.phi_aggregate.n_hidden_layers=1 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.x_dim=20 datamodule.phi_dim=20 datamodule.y_dim=20 datamodule.datasets.train.seq_min_length=10 datamodule.datasets.train.seq_max_length=10

# Evaluation

CKPT="path/to/run_dir/checkpoints/last.ckpt"

python3 run_inference.py +experiment/inference=inference datamodule=rnn datamodule.mixing_architecture.load=True trainer.accelerator="cpu" trainer.devices="auto" datamodule.dataset_parameters.batch_size=256 "model.checkpoint_path='$CKPT'" datamodule.datasets.test.seq_min_length=100
```
