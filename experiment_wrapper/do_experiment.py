import multiprocessing as mp
import os
import platform

import hydra
import omegaconf
import torch
import wandb
from hydra.core.config_store import ConfigStore

from experiment_wrapper.structured_config import Config, DatasetConfig
from tc_transfer.main import main

cs = ConfigStore.instance()
cs.store(name="base_config_class", node=Config)
cs.store(group="dataset", name="base_dataset", node=DatasetConfig)

def flatten_one_level(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                result[f"{k}.{sub_k}"] = sub_v
        else:
            result[k] = v
    return result


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def do_experiment(config: Config):
    if config.print_configuration:
        print("Running experiment with config:")
        print(omegaconf.OmegaConf.to_yaml(config, resolve=True))
    dict_conf: dict =  omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True, enum_to_str=True) # type: ignore
    dict_conf.pop("wandb")
    dict_conf = flatten_one_level(dict_conf)
    wandb.init(
        entity=config.wandb.project.split("/")[0],
        project=config.wandb.project.split("/")[1],
        dir=config.temp_dir,
        tags=config.wandb.tags + [config.dataset.name],
        config=dict_conf,
        reinit="finish_previous",
    )
    wandb.config.update({
        "mlp_hidden_sizes_repr": str(config.mlp_hidden_sizes),
        "splits": str(config.splits),
    }, allow_val_change=True)
    # This is needed for MetaCentrum runs, otherwise the process can take all GPU memory and is slowed down
    torch.cuda.set_per_process_memory_fraction(0.90)
    # Convert log scale parameters to linear scale
    if config.lr != 0:
        config.lr = 10 ** config.lr
    if config.weight_decay != 0:
        config.weight_decay = 10 ** config.weight_decay
    # Create dirs for preloading data, results, and trained models
    os.makedirs(os.path.join(config.temp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(config.temp_dir, "preload"), exist_ok=True)
    os.makedirs(os.path.join(config.temp_dir, "models"), exist_ok=True)

    main(config)

if __name__ == "__main__":
    if platform.system() == "Windows":
        mp.set_start_method("spawn")
    do_experiment()
