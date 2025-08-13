import math
import os
import warnings

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from experiment_wrapper.structured_config import Config, EmbedderFeatures, EmbedderFinetuning
from tc_transfer.finetune_utils.regularization import LDIFSRegularization, SPRegularization
from tc_transfer.model_utils import DatasetWithTransform, load_30pktTCNET_256

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


def cosine_annealing_coef(epoch, T_max, max_val, min_val=0):
    return min_val + 0.5 * (max_val - min_val) * (1 + math.cos(math.pi * epoch / T_max))

def set_dropout_eval(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        m.eval()

def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        # Freeze running mean and std:
        m.eval()
        # # Freeze parameters:
        for p in m.parameters():
            p.requires_grad = False

def setup_adamw_optimizer(config: Config, embedding_model: nn.Module, classification_head: nn.Module) -> optim.Optimizer:
    head_dec_params = []
    head_nodec_params = []
    for name, param in classification_head.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            head_nodec_params.append(param)
        else:
            head_dec_params.append(param)
    parameters = [
        {"params": head_dec_params, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": head_nodec_params, "lr": config.lr, "weight_decay": 0}
    ]
    if config.embedder_finetuning == EmbedderFinetuning.STANDARD: # All embedding model parameters are finetunig with the same learning rate
        embedder_dec_params =[]
        embedder_nodec_params = []
        for name, param in embedding_model.named_parameters():
            if param.ndim <= 1 or name.endswith(".bias") or "embedding" in name:
                embedder_nodec_params.append(param)
            else:
                embedder_dec_params.append(param)
        parameters.append({"params": embedder_dec_params, "lr": config.lr, "weight_decay": config.weight_decay})
        parameters.append({"params": embedder_nodec_params, "lr": config.lr, "weight_decay": 0})
    elif config.embedder_finetuning == EmbedderFinetuning.LAYERWISE_LR:
        # First "block" of parameters is the neck, MLP shared and global pooling, which are trained with config.lr
        neck_params = list(embedding_model.fc.named_parameters()) + list(embedding_model.bn.named_parameters())
        mlp_shared_params = list(embedding_model.backbone_model.mlp_shared.named_parameters())
        global_pool_params = list(embedding_model.backbone_model.cnn_ppi_global_pool.named_parameters())
        param_list_dec = []
        param_list_nodec = []
        for name, param in neck_params + mlp_shared_params + global_pool_params:
            if param.ndim <= 1 or name.endswith(".bias"):
                param_list_nodec.append(param)
            else:
                param_list_dec.append(param)
        parameters.append({"params": param_list_dec, "lr": config.lr, "weight_decay": config.weight_decay})
        parameters.append({"params": param_list_nodec, "lr": config.lr, "weight_decay": 0})
        # Then the four CNN blocks of the backbone model are trained with smaller learning rates config.lr * (config.layerwise_lr_mult ** i)
        cnn_blocks = [-1, -2, -3, -4]
        for i, b in enumerate(cnn_blocks, start=1):
            param_list_dec = []
            param_list_nodec = []
            for name, param in embedding_model.backbone_model.cnn_ppi[b].named_parameters():
                if param.ndim <= 1 or name.endswith(".bias"):
                    param_list_nodec.append(param)
                else:
                    param_list_dec.append(param)
            parameters.append({"params": param_list_dec, "lr": config.lr * (config.layerwise_lr_mult ** i), "weight_decay": config.weight_decay})
            parameters.append({"params": param_list_nodec, "lr": config.lr * (config.layerwise_lr_mult ** i), "weight_decay": 0})
        # At last, the packet size and IPT embeddings are trained with the smallest learning rate
        packet_embedding_params = list(embedding_model.backbone_model.packet_size_nn_embedding.parameters()) + \
                                  list(embedding_model.backbone_model.packet_ipt_nn_embedding.parameters())
        parameters.append({"params": packet_embedding_params, "lr": config.lr * (config.layerwise_lr_mult ** (len(cnn_blocks) + 1)), "weight_decay": 0,})
    optimizer = optim.AdamW(parameters)
    return optimizer

def validate_model(
    classification_head: nn.Module,
    embedding_model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn,
    device: torch.device,
) -> tuple[float, float]:
    assert isinstance(val_dataloader.dataset, DatasetWithTransform)
    embedding_model.eval()
    classification_head.eval()
    epoch_val_loss = 0
    num_samples = len(val_dataloader.dataset)
    true_labels = []
    preds = []
    with torch.no_grad():
        for batch_ppi, batch_labels in val_dataloader:
            batch_ppi, batch_labels = batch_ppi.to(device), batch_labels.to(device)
            batch_embeddings = embedding_model(batch_ppi)
            out = classification_head(batch_embeddings)
            loss = loss_fn(out, batch_labels)
            epoch_val_loss += loss.item() * len(batch_labels)
            batch_preds = out.argmax(dim=1)
            true_labels.append(batch_labels)
            preds.append(batch_preds)
    true_labels, preds = torch.cat(true_labels).cpu().numpy(), torch.cat(preds).cpu().numpy()
    acc = (true_labels == preds).mean()
    epoch_val_loss /= num_samples
    return acc, epoch_val_loss

def train_classification_head_and_finetune_embedding_model(
    train_dataset: DatasetWithTransform,
    val_dataset: DatasetWithTransform,
    embedding_model: nn.Module,
    classification_head: nn.Module,
    config: Config,
    device: torch.device,
    silent: bool = False,
) -> float:
    """"""
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    epoch_num_samples = len(train_dataloader) * config.batch_size
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=2048,
        shuffle=False,
        drop_last=False,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = setup_adamw_optimizer(
        config=config,
        embedding_model=embedding_model,
        classification_head=classification_head,
    )
    if hasattr(embedding_model.backbone_model.cnn_ppi_global_pool[0], "p") and not config.embedder_features == EmbedderFeatures.CNN_BACKBONE_PLUS_GEM_POOLING_LEARNABLE:
        embedding_model.backbone_model.cnn_ppi_global_pool[0].p.requires_grad = False
    # Start point and feature space regularization
    if config.start_point_reg_alpha > 0 or config.feature_space_reg_alpha > 0:
        orig_embedding_model, _ = load_30pktTCNET_256(config=config, device=device, silent=True)
        if config.start_point_reg_alpha > 0:
            start_point_reg_fn = SPRegularization(
                source_model=orig_embedding_model,
                target_model=embedding_model,
            )
        if config.feature_space_reg_alpha:
            feature_space_reg_fn = LDIFSRegularization(
                source_model=orig_embedding_model,
                target_model=embedding_model,
            )
    warmup_iters = round(config.warmup_epochs * len(train_dataloader))
    cosince_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.num_epochs * len(train_dataloader)) - warmup_iters)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_iters)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosince_scheduler], [warmup_iters])
    best_val_acc = 0.0
    best_epoch = 0
    if not silent: print(f"Started training of the embedding model")
    for epoch in range(config.num_epochs):
        classification_head.train()
        embedding_model.train()
        if config.embedder_dropout_eval_mode:
            embedding_model.apply(set_dropout_eval)
        if config.embedder_batchnorm_eval_mode:
            embedding_model.apply(set_batchnorm_eval)
        epoch_train_loss = epoch_xe_loss = epoch_sp_reg = epoch_fs_reg = 0
        for batch_ppi, batch_labels in train_dataloader:
            batch_ppi, batch_labels = batch_ppi.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            batch_embeddings = embedding_model(batch_ppi)
            out = classification_head(batch_embeddings)
            xe_loss = loss_fn(out, batch_labels)
            feature_space_reg = start_point_reg = 0.0
            if config.feature_space_reg_alpha > 0:
                feature_space_reg = config.feature_space_reg_alpha * feature_space_reg_fn(batch_ppi) # type: ignore[reportUnknownVariableType]
            if config.start_point_reg_alpha > 0:
                start_point_reg = config.start_point_reg_alpha * start_point_reg_fn() # type: ignore[reportUnknownVariableType]
            loss = xe_loss + feature_space_reg + start_point_reg
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item() * len(batch_labels)
            epoch_xe_loss += xe_loss.item() * len(batch_labels)
            epoch_fs_reg += feature_space_reg * len(batch_labels)
            epoch_sp_reg += start_point_reg * len(batch_labels)
        epoch_train_loss /= epoch_num_samples
        epoch_xe_loss /= epoch_num_samples
        epoch_sp_reg /= epoch_num_samples
        epoch_fs_reg /= epoch_num_samples
        val_acc, epoch_val_loss = validate_model(
            classification_head=classification_head,
            embedding_model=embedding_model,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
        )
        if epoch >= config.num_epochs // 2 and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(classification_head.state_dict(), os.path.join(config.temp_dir, "best_epoch_head.pt"))
            torch.save(embedding_model.state_dict(), os.path.join(config.temp_dir, "best_epoch_embedding_model.pt"))
        if config.early_stopping_patience > 0 and epoch - best_epoch >= config.early_stopping_patience:
            if not silent: print(f"\tValidation performance did not improve for {config.early_stopping_patience} epochs. Stopping neural network training.")
            break
        if not silent:
            print(
                f"\tEpoch {epoch + 1}/{config.num_epochs}, train-loss: {epoch_train_loss:.4f} "
                f"(xe: {epoch_xe_loss:.4f}, sp: {epoch_sp_reg:.4f}, fs: {epoch_fs_reg:.4f}), " # type: ignore[reportUnknownVariableType]
                f"val-loss: {epoch_val_loss:.4f}, val-acc: {val_acc:.4f}"
            )
    if not silent: print(f"\tReturning the best model from epoch {best_epoch + 1} with val-acc {best_val_acc:.4f}")
    # Load the best model weights
    classification_head.load_state_dict(torch.load(os.path.join(config.temp_dir, "best_epoch_head.pt"), weights_only=True))
    embedding_model.load_state_dict(torch.load(os.path.join(config.temp_dir, "best_epoch_embedding_model.pt"), weights_only=True))
    return best_val_acc
