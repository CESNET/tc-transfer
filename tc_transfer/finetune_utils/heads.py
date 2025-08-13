from typing import Callable, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F

from experiment_wrapper.structured_config import Config, HeadType, NormLayer, TransferMethod
from tc_transfer.model_utils import compute_embeddings


class LinearHead(nn.Module):
    def __init__(self, linear: torch.nn.Linear, dropout: float = 0.0, normalize: bool = False,):
        super().__init__()
        self.normalize = normalize
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear = linear

    def forward(self, input):
        out = input
        if self.normalize:
            out = F.normalize(input)
        out = self.dropout(out)
        out = self.linear(out)
        return out

class MLPHead(torch.nn.Sequential):
    """
    Modified implementation of MLP from torchvision
        - no dropout after the last layer
    https://docs.pytorch.org/vision/main/generated/torchvision.ops.MLP.html
    """
    def __init__(
        self,
        in_size: int,
        hidden_sizes: tuple[int, ...],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.LayerNorm,
        activation_fn: Callable[..., torch.nn.Module] = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        layers = []
        in_dim = in_size
        for hidden_dim in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_fn())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_sizes[-1], bias=bias))
        super().__init__(*layers)

def fit_linear(
    train_embeddings,
    train_labels,
    device,
    num_classes: int,
    config: Config,
    max_epochs: int = 2000,
    lr: float = 0.1
):
    embedding_size = train_embeddings.shape[1]
    linear_model = LinearHead(
        linear=nn.Linear(embedding_size, num_classes),
        normalize=config.head_normalize,
    ).to(device)
    linear_model.train()
    optimizer = torch.optim.AdamW(params=linear_model.parameters(), lr=lr)
    for it in range(max_epochs):
        optimizer.zero_grad()
        logits = linear_model(train_embeddings)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
    return linear_model

def fit_mlp(
    train_embeddings,
    train_labels,
    device,
    config: Config,
    max_epochs=2000,
    lr=1.0,
):
    num_labels = torch.unique(train_labels).shape[0]
    embedding_size = train_embeddings.shape[1]
    mlp_head = MLPHead(
        in_size=embedding_size,
        hidden_sizes=config.mlp_hidden_sizes + (num_labels,),
        norm_layer=nn.LayerNorm if config.mlp_norm_layer == NormLayer.LAYER_NORM else nn.BatchNorm1d if config.mlp_norm_layer == NormLayer.BATCH_NORM else None,
        dropout=config.head_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(params=mlp_head.parameters(), lr=lr)
    mlp_head.train()
    for _ in range(max_epochs):
        optimizer.zero_grad()
        logits = mlp_head(train_embeddings)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
    return mlp_head

def do_linear_probe_sklearn(
    train_embeddings,
    train_labels,
    small_train_embeddings,
    small_train_labels,
    val_embeddings,
    val_labels,
    num_classes: int,
    device,
    silent: bool = False,
):
    train_embeddings = F.normalize(train_embeddings).cpu().numpy()
    small_train_embeddings = F.normalize(small_train_embeddings).cpu().numpy()
    val_embeddings = F.normalize(val_embeddings).cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    small_train_labels = small_train_labels.cpu().numpy()
    val_labels = val_labels.cpu().numpy()
    embedding_size = train_embeddings.shape[1]

    C_regs = [10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
    best_score = 0
    best_C = C_regs[0]
    for c in C_regs:
        logistic_reg = LogisticRegression(C=c, max_iter=1000, verbose=2).fit(small_train_embeddings, small_train_labels)
        score = logistic_reg.score(val_embeddings, val_labels)
        if not silent:
            print(f"Logistic regression with C={c} achieved val-acc {score * 100:.2f}%")
        if (score > best_score):
            best_score = score
            best_C = c
    if not silent:
        print(f"Learning rate sweep for linear probing: best C {best_C} with val-acc {best_score * 100:.2f}%")
    best_logistic_reg = LogisticRegression(C=best_C, max_iter=1000, verbose=True).fit(train_embeddings, train_labels)
    if num_classes == 2 and best_logistic_reg.coef_.shape[0] == 1:
        # Special case for binary classification
        weight = torch.tensor(np.vstack([
            -best_logistic_reg.coef_,
            best_logistic_reg.coef_,
        ]), dtype=torch.float32)
        bias = torch.tensor(np.vstack([
            -best_logistic_reg.intercept_,
            best_logistic_reg.intercept_,
        ]).squeeze(), dtype=torch.float32)
    else:
        weight = torch.tensor(best_logistic_reg.coef_, dtype=torch.float32)
        bias = torch.tensor(best_logistic_reg.intercept_, dtype=torch.float32)

    nn_linear = nn.Linear(in_features=embedding_size, out_features=num_classes)
    nn_linear.weight.data = weight
    nn_linear.bias.data = bias
    classification_head = LinearHead(
        linear=nn_linear,
        normalize=True,
    ).to(device)
    return classification_head

def do_linear_probe(
    train_embeddings,
    train_labels,
    small_train_embeddings,
    small_train_labels,
    val_embeddings,
    val_labels,
    num_classes: int,
    config: Config,
    device,
    silent: bool = False,
):
    """
    Linear probing function using pytorch which first does a hyperparameter sweep and
    then selects the best model and trains it on the full training set.
    """
    lrs = [0.001, 0.0025, 0.05, 0.075, 0.1, 0.2]
    best_score = 0
    best_lr = lrs[0]
    for lr in lrs:
        linear_model = fit_linear(
            small_train_embeddings,
            small_train_labels,
            num_classes=num_classes,
            config=config,
            device=device,
            lr=lr,
        )
        linear_model.eval()
        with torch.no_grad():
            preds = linear_model(val_embeddings).argmax(dim=-1)
            score = (preds == val_labels).sum() / len(val_labels)
        if (score > best_score):
            best_score = score
            best_lr = lr
    if not silent:
        print(f"Learning rate sweep for linear probing: best lr {best_lr} with val-acc {best_score * 100:.2f}%")
    # Train the linear model on the full training set with the best learning rate
    return fit_linear(
        train_embeddings,
        train_labels,
        num_classes=num_classes,
        config=config,
        device=device,
        lr=best_lr,
    )

def do_mlp_probe(
    train_embeddings,
    train_labels,
    small_train_embeddings,
    small_train_labels,
    val_embeddings,
    val_labels,
    config: Config,
    device,
    silent: bool = False,
):
    lrs = [0.001, 0.0025, 0.05, 0.075, 0.1, 0.2]
    best_score = 0
    best_lr = lrs[0]

    for lr in lrs:
        mlp_model = fit_mlp(small_train_embeddings, small_train_labels, config=config, device=device, lr=lr)
        mlp_model.eval()
        with torch.no_grad():
            preds = mlp_model(val_embeddings).argmax(dim=-1)
            score = (preds == val_labels).sum() / len(val_labels)
        if (score > best_score):
            best_score = score
            best_lr = lr
    if not silent:
        print(f"MLP training best LR: {best_lr}, best val-acc: {best_score * 100:.2f}%")
    return fit_mlp(
        train_embeddings,
        train_labels,
        config=config,
        device=device,
        lr=best_lr,
    )

def build_classification_head(
    embedding_model,
    train_dataset,
    val_dataset,
    config: Config,
    device,
    silent: bool = False,
):
    train_embeddings = compute_embeddings(embedding_model, train_dataset, device)
    train_labels = train_dataset.encoded_labels.to(device)
    num_classes = len(train_dataset.encoder.classes_)
    embedding_size = train_embeddings.shape[1]
    if config.transfer_method == TransferMethod.FROM_SCRATCH:
        # When training from scratch, just return head with random weights
        if config.head == HeadType.LINEAR:
            classification_head = LinearHead(
                linear=nn.Linear(embedding_size, num_classes),
                dropout=config.head_dropout,
                normalize=config.head_normalize,
            ).to(device)
        elif config.head == HeadType.MLP:
            classification_head = MLPHead(
                in_size=embedding_size,
                hidden_sizes=config.mlp_hidden_sizes + (num_classes,),
                norm_layer=nn.LayerNorm if config.mlp_norm_layer == NormLayer.LAYER_NORM else nn.BatchNorm1d if config.mlp_norm_layer == NormLayer.BATCH_NORM else None,
                dropout=config.head_dropout,
            ).to(device)
        else:
            raise ValueError(f"Unsupported head: {config.head}")
        return classification_head
    val_embeddings = compute_embeddings(embedding_model, val_dataset, device)
    val_labels = val_dataset.encoded_labels.to(device)
    if config.transfer_method == TransferMethod.FINETUNE or config.dataset.name == "CESNET-TLS22" or config.dataset.name == "AppClassNet":
        # Use smaller datasets to speed up the linear probing on large datasets or when finetuning an embedding model
        train_embeddings = train_embeddings[:100_000]
        train_labels = train_labels[:100_000]
        val_embeddings = val_embeddings[:25_000]
        val_labels = val_labels[:25_000]
    # Subset of the training set used for a learning rate sweep
    small_train_embeddings = train_embeddings[:10_000]
    small_train_labels = train_labels[:10_000]
    # Linear probing
    if config.head == HeadType.LINEAR:
        if config.linear_probe_exact_solver:
            classification_head = do_linear_probe_sklearn(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                small_train_embeddings=small_train_embeddings,
                small_train_labels=small_train_labels,
                val_embeddings=val_embeddings,
                val_labels=val_labels,
                num_classes=num_classes,
                device=device,
                silent=silent,
            )
        else: # Otherwise use a pytorch training loop with nn.Linear
            classification_head = do_linear_probe(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                small_train_embeddings=small_train_embeddings,
                small_train_labels=small_train_labels,
                val_embeddings=val_embeddings,
                val_labels=val_labels,
                num_classes=num_classes,
                config=config,
                device=device,
                silent=silent,
            )
        if config.transfer_method == TransferMethod.FINETUNE:
            # Linear probing does not use dropout, but we want to add it to the returned classification head
            classification_head = LinearHead(
                linear=classification_head.linear,
                dropout=config.head_dropout,
                normalize=config.head_normalize,
            ).to(device)
    elif config.head == HeadType.MLP:
        classification_head = do_mlp_probe(
            small_train_embeddings=small_train_embeddings,
            small_train_labels=small_train_labels,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            config=config,
            device=device,
            silent=silent,
        )
    else:
        raise ValueError(f"Unsupported head: {config.head}")
    return classification_head
