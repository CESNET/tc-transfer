from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from omegaconf import MISSING


class EmbedderFeatures(Enum):
    ORIGINAL = auto()
    SKIP_NECK = auto()
    CNN_BACKBONE_PLUS_GEM_POOLING = auto()
    CNN_BACKBONE_PLUS_GEM_POOLING_LEARNABLE = auto()
    CNN_BACKBONE_PLUS_MAX_POOLING = auto()
    CNN_BACKBONE_PLUS_AVG_POOLING = auto()

class EmbedderFinetuning(Enum):
    STANDARD = auto()
    LAYERWISE_LR = auto()

class HeadType(Enum):
    LINEAR = auto()
    MLP = auto()

class NormLayer(Enum):
    NONE = auto()
    BATCH_NORM = auto()
    LAYER_NORM = auto()

class TransferMethod(Enum):
    INPUT_SPACE = auto()
    FROM_SCRATCH = auto()
    KNN = auto()
    LINEAR_PROBE = auto()
    FINETUNE = auto()

class DatasetLoader(Enum):
    CSV = auto()
    TCBENCH = auto()
    ISCXVPN2016 = auto()
    CESNET_DATAZOO = auto()
    APPCLASSNET = auto()

@dataclass
class WandbConfig:
    project: str = MISSING
    tags: list[str] = MISSING

@dataclass
class DatasetConfig:
    name: str = MISSING
    loader: DatasetLoader = MISSING
    dataset_path: Optional[str] = None
    train_size: Any = "all"
    val_size: Any = "all"
    test_size: Any = "all"
    label_column: Optional[str] = None
    random_split_val_test_fraction: Optional[float] = None

@dataclass
class Config:
    wandb: WandbConfig = MISSING
    temp_dir: str = MISSING
    print_configuration: bool = False
    dataset: DatasetConfig = MISSING
    dataset_base_dir: str = MISSING
    splits: tuple[int] = MISSING
    faiss_ranking_n: int = 100
    skip_test_evaluation: bool = False
    save_model: bool = False
    transfer_method: TransferMethod = TransferMethod.FINETUNE
    linear_probe_exact_solver: bool = False
    # Finetuning parameters
    batch_size: int = 256
    num_epochs: int = 50
    warmup_epochs: float = 2
    lr: float = -2
    layerwise_lr_mult: float = 0.7
    weight_decay: float = 0
    feature_space_reg_alpha: float = 0
    start_point_reg_alpha: float = 0
    early_stopping_patience: int = -1
    head: HeadType = HeadType.LINEAR
    head_dropout: float = 0.35
    head_normalize: bool = True
    mlp_hidden_sizes: tuple[int, ...] = (256,)
    mlp_norm_layer: NormLayer = NormLayer.BATCH_NORM
    embedder_features: EmbedderFeatures = EmbedderFeatures.ORIGINAL
    embedder_finetuning: EmbedderFinetuning = EmbedderFinetuning.STANDARD
    embedder_batchnorm_eval_mode: bool = True
    embedder_dropout_eval_mode: bool = False
    embedder_replace_unseen_packets_threshold: int = 1
    cross_dataset_transfer: Optional[str] = None # when using other datasets for pretraining, provide path to model weights
