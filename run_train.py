from src.utils import hydra_custom_resolvers
import hydra
from omegaconf import DictConfig
import os

# python run_training.py training=<evaluation_config> run_name=<run_name>

@hydra.main(config_path="configs", config_name="train_root", version_base="1.2")
def main(hydra_config: DictConfig):
   
    import src.utils.general as utils
    from src.training_pipeline import train

    utils.extras(hydra_config)
    print(f"Original working directory: {hydra_config.work_dir}")

    # Train model
    train(hydra_config)


if __name__ == "__main__":
    main()
