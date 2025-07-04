import copy
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import recall_score, precision_score, f1_score

class PruningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        pruning_method: str = "l1_unstructured",
        amount: float = 0.2,
        n_norm: int = 1,
        layers_to_prune: List[str] = None,
        global_pruning: bool = False,
        prune_every_n_epochs: int = 1,
        optimizer: Dict[str, str] = None,
        scheduler: Dict[str, str] = None,
        loss: str = "cross_entropy"
    ):
        """
        PyTorch Lightning module for model pruning.
        
        Args:
            model: The model to prune
            pruning_method: One of ['l1_unstructured', 'l1_structured', 'random_unstructured', 'ln_structured']
            amount: Fraction of connections to prune (0.0-1.0)
            n_norm: Do pruning in n_norm normalize
            layers_to_prune: List of layer names to prune (None = all Conv/Linear layers)
            global_pruning: Whether to prune across all specified layers simultaneously
            prune_every_n_epochs: Frequency of pruning
            optimizer: What optimizer to use for training and its params
            scheduler: Scheduler for optimizer and its params
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = copy.deepcopy(model)
        self.validation_data = []
        self.pruning_config = {
            "method": pruning_method,
            "amount": amount,
            "n_norm": n_norm,
            "layers": layers_to_prune,
            "global": global_pruning,
            "frequency": prune_every_n_epochs,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss": loss
        }
        
        # Initialize pruning parameters
        self._setup_pruning()
        
    def _setup_pruning(self):
        # """Identify layers to prune and initialize pruning containers"""
        self.pruning_config["layers"] = [
            name for name, module in self.model.named_modules()
            if isinstance(module, tuple([getattr(nn, layer) for layer in self.pruning_config["layers"]]))
        ]
    
    def _apply_pruning(self):
        """Apply pruning to all specified layers"""
        current_amount = self.pruning_config["amount"]
        
        if self.pruning_config["global"]:
        
        # Create pruning containers
            parameters_to_prune = []
            for name, module in self.model.named_modules(): 
                if name in self.pruning_config["layers"]:
                    parameters_to_prune.append((module, "weight"))
            
            if "unstructured" in self.pruning_config["method"].lower():
                prune.global_unstructured(
                     parameters_to_prune,
                     pruning_method=getattr(prune, self.pruning_config["method"]),
                     amount=current_amount
                )
            else:
                for name, module in self.model.named_modules(): 
                    if name in self.pruning_config["layers"]:
                        if "unstructured" in self.pruning_config["method"].lower():
                            prune_method = getattr(prune, self.pruning_config["method"])
                            prune_method(module, name="weight", amount=current_amount)
                        else:
                            dim = self.pruning_config["n_norm"]
                            prune.ln_structured(
                                module, name="weight", amount=current_amount, n=dim, dim=1
                            )
        else:
            # Layer-wise pruning
            for name, module in self.model.named_modules(): 
                if name in self.pruning_config["layers"]:
                    if "unstructured" in self.pruning_config["method"].lower():
                        prune_method = getattr(prune, self.pruning_config["method"])
                        prune_method(module, name="weight", amount=current_amount)
                    else:
                        dim = self.pruning_config["n_norm"]
                        prune.ln_structured(
                            module, name="weight", amount=current_amount, n=dim, dim=1
                        )
        
        # Log sparsity statistics
        self._log_sparsity()
    
    def _log_sparsity(self):
        """Log sparsity statistics to TensorBoard"""
        total_params = 0
        total_pruned = 0
        
        for name, module in self.model.named_modules(): 
            if name in self.pruning_config["layers"]:
                sparsity = torch.sum(module.weight == 0).item() / module.weight.numel()
                self.log(f"pruning/sparsity/{name}", sparsity, prog_bar=False)
                total_params += module.weight.numel()
                total_pruned += torch.sum(module.weight == 0).item()
        
        global_sparsity = total_pruned / total_params if total_params > 0 else 0
        self.log("pruning/global_sparsity", global_sparsity, prog_bar=True)
        self.log("Pruned params: ", total_pruned)
    
    def on_train_epoch_start(self):
        """Apply pruning at specified intervals"""
        current_epoch = self.current_epoch
        if (current_epoch % self.pruning_config["frequency"] == 0):
            self._apply_pruning()
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optim = getattr(torch.optim, self.pruning_config["optimizer"]["name"])
        optim = optim(self.parameters(), **self.pruning_config["optimizer"]["params"])

        if self.pruning_config["scheduler"] is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.pruning_config["scheduler"]["name"])
            scheduler = scheduler(optim, **{key: eval(val) if "lambda" in key else val for key, val in self.pruning_config["scheduler"]["params"].items()})

            return [optim], [scheduler]

        return optim

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        optim = getattr(nn.functional, self.pruning_config["loss"])
        loss = optim(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.validation_data.append((torch.argmax(y_hat, -1), y))
        optim = getattr(nn.functional, self.pruning_config["loss"])
        loss = optim(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        model_logits = torch.cat([x[0] for x in self.validation_data]).cpu()
        model_targets = torch.cat([x[1] for x in self.validation_data]).cpu()

        accuracy = (model_logits == model_targets).float().mean().item()
        metrics = {
            'val_recall_micro': recall_score(model_targets, model_logits, average='micro', zero_division=0),
            'val_recall_macro': recall_score(model_targets, model_logits, average='macro', zero_division=0),
            'val_recall_weighted': recall_score(model_targets, model_logits, average='weighted', zero_division=0),

            'val_precision_micro': precision_score(model_targets, model_logits, average='micro', zero_division=0),
            'val_precision_macro': precision_score(model_targets, model_logits, average='macro', zero_division=0),
            'val_precision_weighted': precision_score(model_targets, model_logits, average='weighted', zero_division=0),

            'val_f1_micro': f1_score(model_targets, model_logits, average='micro', zero_division=0),
            'val_f1_macro': f1_score(model_targets, model_logits, average='macro', zero_division=0),
            'val_f1_weighted': f1_score(model_targets, model_logits, average='weighted', zero_division=0),
        }
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log_dict(metrics)
        self.validation_data.clear()

        metrics.update({'val_accuracy': accuracy})
        return metrics
