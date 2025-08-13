import json
import os
from typing import Optional

import numpy as np
import torch
import wandb
from sklearn.preprocessing import LabelEncoder

from experiment_wrapper.structured_config import Config, HeadType, TransferMethod, EmbedderFeatures
from tc_transfer.dataset_utils import load_dataset
from tc_transfer.embeddings_evaluation import (evaluate_classification_head_with_possible_finetune,
                                               evaluate_knn_with_fixed_embeddings,
                                               evaluate_linear_probing_with_fixed_embeddings)
from tc_transfer.input_space_baseline import prepare_input_space_embeddings
from tc_transfer.metrics import METRICS
from tc_transfer.model_utils import DatasetWithTransform, compute_embeddings, load_30pktTCNET_256


def evaluate_dataset(config: Config, device: torch.device) -> tuple[dict[str, float], dict[str, float], Optional[float]]:
    dataset_name = config.dataset.name
    per_split_test_metrics =  []
    per_split_val_acc = []
    print(f"Processing dataset {dataset_name} with splits {config.splits} (printing output for the first split)")
    for split_id in config.splits:
        silent = False # split_id != 0
        train_data, val_data, test_data, train_labels, val_labels, test_labels = load_dataset(
            dataset_name=dataset_name,
            dataset_loader=config.dataset.loader,
            split_id=split_id,
            preload_base=os.path.join(config.temp_dir, "preload"),
            notcb_dataset_path=config.dataset.dataset_path,
            iscxvpn_label_column=config.dataset.label_column,
            random_split_val_test_fraction=config.dataset.random_split_val_test_fraction,
            train_size=config.dataset.train_size,
            val_size=config.dataset.val_size,
            test_size=config.dataset.test_size,
            silent=silent,
        )
        #
        # Evaluate kNN on input-space baseline embeddings
        #
        if config.transfer_method == TransferMethod.INPUT_SPACE:
            baseline_train_embeddings = prepare_input_space_embeddings(train_data, baseline_params={})
            baseline_test_embeddings = prepare_input_space_embeddings(test_data, baseline_params={})
            metrics_tuple = evaluate_knn_with_fixed_embeddings(
                train_embeddings=baseline_train_embeddings,
                train_labels=train_labels,
                test_embeddings=baseline_test_embeddings,
                test_labels=test_labels,
                knn_metric="L1",
                ranking_n=config.faiss_ranking_n,
                device=device,
                silent=silent,
            )
            per_split_test_metrics.append(metrics_tuple)
            print(f"Input space baseline results for split {split_id}: top1-acc {metrics_tuple.top1_acc * 100:.2f}, maj-acc {metrics_tuple.maj_acc * 100:.2f}")
        else:
            # For other transfer methods, load the 30pktTCNET-256 embedding model
            embedding_model, ppi_transform = load_30pktTCNET_256(config=config, device=device)
            # Prepare datasets with the PPI transform and label encoding
            label_encoder = LabelEncoder().fit(train_labels)
            train_dataset = DatasetWithTransform(data=train_data, labels=train_labels, ppi_transform=ppi_transform, encoder=label_encoder)
            val_dataset = DatasetWithTransform(data=val_data, labels=val_labels, ppi_transform=ppi_transform, encoder=label_encoder)
            test_dataset = DatasetWithTransform(data=test_data, labels=test_labels, ppi_transform=ppi_transform, encoder=label_encoder)
            #
            # Evaluate kNN on fixed pretrained embeddings
            #
            if config.transfer_method == TransferMethod.KNN:
                if config.embedder_features != EmbedderFeatures.ORIGINAL:
                    print("KNN transfer method is intended to be used with the original features of 30pktTCNET_256")
                if not silent: print("\tUsing the pretrained 30pktTCNET_256 model to compute embeddings")
                train_embeddings = compute_embeddings(embedding_model, dataset=train_dataset, device=device, return_tensors=False)
                test_embeddings = compute_embeddings(embedding_model, dataset=test_dataset, device=device, return_tensors=False)
                metrics_tuple = evaluate_knn_with_fixed_embeddings(
                    train_embeddings=train_embeddings,
                    train_labels=train_labels,
                    test_embeddings=test_embeddings,
                    test_labels=test_labels,
                    knn_metric="cosine",
                    ranking_n=config.faiss_ranking_n,
                    device=device,
                    silent=silent,
                )
                per_split_test_metrics.append(metrics_tuple)
                print(f"Pretrained model with KNN results for split {split_id}: top1-acc {metrics_tuple.top1_acc * 100:.2f}, maj-acc {metrics_tuple.maj_acc * 100:.2f}")
            #
            # Evaluate linear probing with fixed pretrained embeddings
            #
            elif config.transfer_method == TransferMethod.LINEAR_PROBE:
                assert config.head == HeadType.LINEAR
                metrics_tuple = evaluate_linear_probing_with_fixed_embeddings(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    embedding_model=embedding_model,
                    config=config,
                    device=device,
                    silent=silent,
                )
                per_split_test_metrics.append(metrics_tuple)
                print(f"Linear probing results for split {split_id}: top1-acc {metrics_tuple.top1_acc * 100:.2f}")
            #
            # Evaluate model finetuning or training from scratch
            #
            elif config.transfer_method == TransferMethod.FINETUNE or config.transfer_method == TransferMethod.FROM_SCRATCH:
                metrics_tuple, best_val_acc = evaluate_classification_head_with_possible_finetune(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    embedding_model=embedding_model,
                    config=config,
                    device=device,
                    silent=silent,
                )
                per_split_test_metrics.append(metrics_tuple)
                per_split_val_acc.append(best_val_acc)
                if not config.skip_test_evaluation:
                    if config.transfer_method == TransferMethod.FROM_SCRATCH:
                        print(f"Training from scratch results for split {split_id}: top1-acc {metrics_tuple.top1_acc * 100:.2f}")
                    else:
                        print(f"Finetuned embedding model results for split {split_id}: top1-acc {metrics_tuple.top1_acc * 100:.2f}")
    test_metrics = {metric: np.mean([getattr(a, metric) for a in per_split_test_metrics]).item() for metric in METRICS}
    test_metrics_std = {metric: np.std([getattr(a, metric) for a in per_split_test_metrics]).item() for metric in METRICS}
    if per_split_val_acc:
        average_val_acc = np.mean(per_split_val_acc).item()
    else:
        average_val_acc = None
    return test_metrics, test_metrics_std, average_val_acc

def main(config: Config) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_metrics, test_metrics_std, average_val_acc = evaluate_dataset(config=config, device=device,)
    if average_val_acc is not None:
        wandb.log({"val/acc": average_val_acc})
    if not config.skip_test_evaluation:
        results_path = os.path.join(config.temp_dir, "results", f"{config.dataset.name}-{config.transfer_method.name}{'_pretrained-on-' + config.cross_dataset_transfer if config.cross_dataset_transfer else ''}.json")
        with open(results_path, "w") as f:
            json.dump({
                "splits": tuple(config.splits),
                "test_metrics": test_metrics,
                "test_metrics_std": test_metrics_std,
            }, f, indent=4)
