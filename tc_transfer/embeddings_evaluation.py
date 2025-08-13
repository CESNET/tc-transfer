import os
import time

import torch
from sklearn.metrics import f1_score, recall_score

from experiment_wrapper.structured_config import Config, TransferMethod
from tc_transfer.finetune_utils.finetune import \
    train_classification_head_and_finetune_embedding_model
from tc_transfer.finetune_utils.heads import build_classification_head
from tc_transfer.metrics import MetricsTuple, compute_smart_maj_preds
from tc_transfer.model_utils import DatasetWithTransform, compute_embeddings, find_ranks_faiss


def evaluate_knn_with_fixed_embeddings(
    train_embeddings, train_labels,
    test_embeddings, test_labels,
    device: torch.device,
    knn_metric: str = "cosine",
    ranking_n: int = 100,
    maj_closeness_threshold: float = 0.0,
    maj_zero_branch: bool = False,
    silent: bool = False,
) -> MetricsTuple:
    """"""
    start_time = time.time()
    if not silent: print(f"\tStarting faiss ranking with {knn_metric} metric and N={ranking_n}")
    distances, ranks = find_ranks_faiss(
        vecs=train_embeddings,
        qvecs=test_embeddings,
        N=ranking_n,
        metric=knn_metric,
        batch_size=10_000 if len(test_embeddings) > 100_000 else None,
        silent=silent,
        device=device,
    )
    if not silent: print(f"\tTime elapsed for faiss ranking: {time.time() - start_time:.2f} s")
    # Compute metrics based on the ranking
    # Top1 prediction
    closest_1 = train_labels[ranks[:, 0]]
    # Distance-based maj vote prediction
    maj_vote = compute_smart_maj_preds(
        ranks,
        distances,
        train_labels,
        maj_closeness_threshold=maj_closeness_threshold,
        maj_zero_branch=maj_zero_branch,
    )
    top1_acc = (test_labels == closest_1).mean()
    maj_acc = (test_labels == maj_vote).mean()
    top1_recall = recall_score(test_labels, closest_1, average="macro", zero_division=0)
    maj_recall = recall_score(test_labels, maj_vote, average="macro", zero_division=0)
    top1_f1 = f1_score(test_labels, closest_1, average="weighted", zero_division=0)
    maj_f1 = f1_score(test_labels, maj_vote, average="weighted", zero_division=0)
    return MetricsTuple(
        top1_acc=top1_acc,
        top1_recall=top1_recall,
        top1_f1=top1_f1,
        maj_acc=maj_acc,
        maj_recall=maj_recall,
        maj_f1=maj_f1,
    )

def evaluate_linear_probing_with_fixed_embeddings(
    train_dataset: DatasetWithTransform,
    val_dataset: DatasetWithTransform,
    test_dataset: DatasetWithTransform,
    embedding_model: torch.nn.Module,
    config: Config,
    device: torch.device,
    silent: bool = False,
) -> MetricsTuple:
    """"""
    classification_head = build_classification_head(
        embedding_model=embedding_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        silent=silent,
    )
    test_embeddings = compute_embeddings(
        embedding_model=embedding_model,
        dataset=test_dataset,
        device=device,
    )
    test_labels = test_dataset.labels
    classification_head.eval()
    embedding_model.eval()
    with torch.no_grad():
        closest_1 = classification_head(test_embeddings).argmax(dim=1).cpu().numpy()
        closest_1 = train_dataset.encoder.inverse_transform(closest_1)
        top1_acc = (test_labels == closest_1).mean()
    top1_recall = recall_score(test_labels, closest_1, average="macro", zero_division=0)
    top1_f1 = f1_score(test_labels, closest_1, average="weighted", zero_division=0)
    return MetricsTuple(
            top1_acc=top1_acc,
            top1_recall=top1_recall,
            top1_f1=top1_f1,
            maj_acc=0.0,
            maj_recall=0.0,
            maj_f1=0.0,
    )

def evaluate_classification_head_with_possible_finetune(
    train_dataset: DatasetWithTransform,
    val_dataset: DatasetWithTransform,
    test_dataset: DatasetWithTransform,
    embedding_model: torch.nn.Module,
    config: Config,
    device: torch.device,
    silent: bool = False,
) -> tuple[MetricsTuple, float]:
    """"""
    classification_head = build_classification_head(
        embedding_model=embedding_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        silent=silent,
    )
    best_val_acc = train_classification_head_and_finetune_embedding_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        embedding_model=embedding_model,
        classification_head=classification_head,
        config=config,
        device=device,
        silent=silent,
    )
    if config.save_model:
        if config.transfer_method != TransferMethod.FROM_SCRATCH:
            raise RuntimeError("Saving of embedding models is intended for models trained from scratch (head is not saved)")
        torch.save(embedding_model.state_dict(), os.path.join(config.temp_dir, "models", f"{config.dataset.name}.pt"))
    if config.skip_test_evaluation:
        return MetricsTuple(top1_acc=0, top1_recall=0, top1_f1=0, maj_acc=0, maj_recall=0, maj_f1=0), best_val_acc
    test_embeddings = compute_embeddings(
        embedding_model=embedding_model,
        dataset=test_dataset,
        device=device,
    )
    test_labels = test_dataset.labels
    classification_head.eval()
    embedding_model.eval()
    with torch.no_grad():
        closest_1 = classification_head(test_embeddings).argmax(dim=1).cpu().numpy()
        closest_1 = train_dataset.encoder.inverse_transform(closest_1)
        top1_acc = (test_labels == closest_1).mean()
    top1_recall = recall_score(test_labels, closest_1, average="macro", zero_division=0)
    top1_f1 = f1_score(test_labels, closest_1, average="weighted", zero_division=0)
    return (
        MetricsTuple(
            top1_acc=top1_acc,
            top1_recall=top1_recall,
            top1_f1=top1_f1,
            maj_acc=0.0,
            maj_recall=0.0,
            maj_f1=0.0,
      ),
      best_val_acc
    )
