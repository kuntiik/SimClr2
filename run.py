import hydra
from omegaconf import DictConfig 
# from src.utils import utils

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config : DictConfig):
    from src.train import train
    # utils.extras(config)
    return train(config)


if __name__ == "__main__":
    main()