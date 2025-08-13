import os
from types import MethodType
from typing import Callable, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from cesnet_models.models import Model_30pktTCNET_256_Weights, model_30pktTCNET_256
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment_wrapper.structured_config import Config, EmbedderFeatures, TransferMethod


class DatasetWithTransform(Dataset):
    ppi_transform: Callable
    encoder: LabelEncoder
    data: torch.Tensor
    labels: np.ndarray
    encoded_labels: torch.Tensor

    def __init__(self, data: np.ndarray, labels: np.ndarray, ppi_transform: Callable, encoder: LabelEncoder) -> None:
        assert len(data) == len(labels)
        self.ppi_transform = ppi_transform
        self.encoder = encoder
        self.data = torch.from_numpy(self.ppi_transform(data).astype("float32"))
        self.labels = labels
        self.encoded_labels = torch.from_numpy(self.encoder.transform(labels)).long()

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.encoded_labels[index]

    def __len__(self) -> int:
        return len(self.labels)

def compute_embeddings(
    embedding_model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    return_tensors: bool = True
) -> np.ndarray | torch.Tensor:
    dataloader = DataLoader(dataset, batch_size=2048, drop_last=False)
    embeddings = []
    embedding_model.eval()
    with torch.no_grad():
        for batch_ppi, _ in dataloader:
            batch_ppi = batch_ppi.to(device)
            batch_embeddings = embedding_model(batch_ppi)
            embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings)
    if return_tensors:
        return embeddings
    else:
        return embeddings.cpu().numpy()

def find_ranks_faiss(vecs,
                     qvecs,
                     device: torch.device,
                     metric: str = "cosine",
                     N: int = 100,
                     batch_size: Optional[int] = None,
                     silent: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if metric == "cosine":
        index = faiss.IndexFlatIP(vecs.shape[-1])
    elif metric == "L1":
        index = faiss.IndexFlat(vecs.shape[-1], faiss.METRIC_L1)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    if device.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
        torch.cuda.empty_cache()
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    index.add(vecs) # type: ignore
    if batch_size is None:
        scores, ranks = index.search(qvecs, N) # type: ignore
    else:
        num_batches = (len(qvecs) // batch_size) + 1
        scores_list = []
        ranks_list = []
        for batch in tqdm(np.array_split(qvecs, num_batches), total=num_batches, disable=silent):
            scores, ranks = index.search(batch, N) # type: ignore
            scores_list.append(scores)
            ranks_list.append(ranks)
        scores = np.concatenate(scores_list, axis=0)
        ranks = np.concatenate(ranks_list, axis=0)
    return scores, ranks

def replace_unseen_packet_embeddings(
    embedding_model,
    replace_threshold: int = 1,
    small_packets_replace_with: int = 0,
    silent: bool = True,
) -> None:
    backbone_model = embedding_model.backbone_model
    if not hasattr(backbone_model, "psizes_hist"):
        print("Histogram of training packet sizes is not available")
        return
    df_train_packets = pd.DataFrame(backbone_model.psizes_hist, columns=["Count"])
    df_train_packets["Perc"] = df_train_packets["Count"] / df_train_packets["Count"].sum()
    packets_to_replace = df_train_packets[df_train_packets["Count"] < replace_threshold].index
    if len(packets_to_replace) == 0:
        print(f"All packet sizes were seen at least {replace_threshold} times")
        return
    # Small <100 unseen packets are replaced with the embedding of 'small_packets_replace_with'
    for i in packets_to_replace[packets_to_replace < 100]: # type: ignore
        backbone_model.packet_size_nn_embedding.weight.data[i] = backbone_model.packet_size_nn_embedding.weight.data[small_packets_replace_with]
        if not silent: print(f"Setting the packet size embedding of {i} ({df_train_packets.Count.iloc[i]} obs) to {small_packets_replace_with} ({df_train_packets.Count.iloc[small_packets_replace_with]} obs)")
    # Big >=1250 unseen packets are replaced with their closest seen packet
    seen_big_packets = [i for i in range(1250, 1501) if i not in packets_to_replace]
    for i in packets_to_replace[packets_to_replace >= 1250]: # type: ignore
        replace_with = min(seen_big_packets, key=lambda x: abs(x - i)) # type: ignore
        if not silent: print(f"Setting the packet size embedding of {i} ({df_train_packets.Count.iloc[i]} obs) to {replace_with} ({df_train_packets.Count.iloc[replace_with]} obs)")
        backbone_model.packet_size_nn_embedding.weight.data[i] = backbone_model.packet_size_nn_embedding.weight.data[replace_with]

def load_30pktTCNET_256(
    config: Config,
    device: torch.device,
    silent: bool = False
) -> tuple[nn.Module, Callable]:
    """"""
    pretrained_weights = Model_30pktTCNET_256_Weights.DEFAULT
    ppi_transform = pretrained_weights.transforms["ppi_transform"]
    if config.transfer_method == TransferMethod.FROM_SCRATCH:
        embedding_model = model_30pktTCNET_256(weights=None)
    else:
        embedding_model = model_30pktTCNET_256(weights=pretrained_weights)
    if config.embedder_replace_unseen_packets_threshold > 0:
        replace_unseen_packet_embeddings(
            embedding_model,
            replace_threshold=config.embedder_replace_unseen_packets_threshold,
        )
    if (
        config.embedder_features == EmbedderFeatures.SKIP_NECK or
        config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_GEM_POOLING or
        config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_GEM_POOLING_LEARNABLE or
        config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_MAX_POOLING or
        config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_AVG_POOLING
    ):
        def skip_neck_forward(self, ppi):
            out = self.backbone_model.forward_features(ppi=ppi, flowstats=None)
            if config.transfer_method == TransferMethod.KNN:
                # Make sure the output is normalized for KNN
                # For other methods, utilizing classification head, normalization is configurable with head_normalize
                out = F.normalize(out)
            return out
            
        embedding_model.forward = MethodType(skip_neck_forward, embedding_model)
        embedding_model.fc = nn.Identity() # type: ignore
        embedding_model.bn = nn.Identity() # type: ignore
    if (
        config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_GEM_POOLING or
        config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_GEM_POOLING_LEARNABLE
       ):
        embedding_model.backbone_model.mlp_shared = nn.Identity()
    elif config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_MAX_POOLING:
        embedding_model.backbone_model.mlp_shared = nn.Identity()
        embedding_model.backbone_model.cnn_ppi_global_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(start_dim=1),
        )
    elif config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_AVG_POOLING:
        embedding_model.backbone_model.mlp_shared = nn.Identity()
        embedding_model.backbone_model.cnn_ppi_global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )
    # If testing cross-dataset transfer (weights pretrained on individual datasets), load the pretrained weights from a local path
    # Do this only after model modificataions based on embedder_features
    if config.cross_dataset_transfer is not None:
        pretrained_weights_path = os.path.join(config.temp_dir, "models", f"{config.cross_dataset_transfer}.pt")
        if not os.path.exists(pretrained_weights_path):
            raise RuntimeError(f"Pretrained weights for cross-dataset transfer not found: {pretrained_weights_path}")
        if not silent:
            print(f"Loading pretrained weights for cross-dataset transfer from {pretrained_weights_path}")
        embedding_model.load_state_dict(torch.load(pretrained_weights_path, weights_only=True))

    embedding_model = embedding_model.to(device)
    return embedding_model, ppi_transform
