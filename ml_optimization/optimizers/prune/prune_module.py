import copy
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class PruningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        pruning_method: str = "l1_unstructured",
        amount: float = 0.2,
        layers_to_prune: List[str] = None,
        global_pruning: bool = False,
        prune_every_n_epochs: int = 1,
        optimizer: Dict[str, str] = None,
        scheduler: Dict[str, str] = None
    ):
        """
        PyTorch Lightning module for model pruning.
        
        Args:
            model: The model to prune
            pruning_method: One of ['l1_unstructured', 'l1_structured', 'random_unstructured', 'ln_structured']
            amount: Fraction of connections to prune (0.0-1.0)
            layers_to_prune: List of layer names to prune (None = all Conv/Linear layers)
            global_pruning: Whether to prune across all specified layers simultaneously
            prune_every_n_epochs: Frequency of pruning
            optimizer: What optimizer to use for training and its params
            scheduler: Scheduler for optimizer and its params
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = copy.deepcopy(model)
        self.pruning_config = {
            "method": pruning_method,
            "amount": amount,
            "layers": layers_to_prune,
            "global": global_pruning,
            "frequency": prune_every_n_epochs,
            "optimizer": optimizer,
            "scheduler": scheduler
        }
        
        # Initialize pruning parameters
        self._setup_pruning()
        
    def _setup_pruning(self):
        # """Identify layers to prune and initialize pruning containers"""
        self.pruning_config["layers"] = [
            name for name, module in self.model.named_modules()
            if isinstance(module, tuple([getattr(nn, layer) for layer in self.pruning_config["layers"]]))
        ]

        print("Before")
        print(self.pruning_config["layers"])
        
        # Create pruning containers
        self.prune_parameters = []
        for name in self.pruning_config["layers"]:
            module = dict(self.model.named_modules())[name]
            
            if "unstructured" in self.pruning_config["method"]:
                prune_method = getattr(prune, self.pruning_config["method"])
                prune_method(module, name="weight", amount=0)  # Initialize with 0 pruning
            elif "structured" in self.pruning_config["method"]:
                dim = int(self.pruning_config["method"].split("_")[0][1:])  # Extract 'l1' -> 1
                prune.ln_structured(module, name="weight", amount=0, n=dim, dim=1)
            
            # Make pruning permanent (remove reparametrization)
            prune.remove(module, "weight")
            self.prune_parameters.append((name, module))
        
        print("After")
        print(self.prune_parameters)
    
    def _apply_pruning(self):
        """Apply pruning to all specified layers"""
        current_amount = self.pruning_config["amount"]
        
        if self.pruning_config["global"]:
            # Global pruning across all layers
            parameters_to_prune = [
                (module, "weight") for _, module in self.prune_parameters
            ]
            
            if "unstructured" in self.pruning_config["method"]:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=getattr(prune, self.pruning_config["method"]),
                    amount=current_amount,
                )
            else:
                dim = int(self.pruning_config["method"].split("_")[0][-1])
                prune.global_structured(
                    parameters_to_prune,
                    pruning_method=prune.LnStructured,
                    amount=current_amount,
                    n=dim,
                    dim=1,
                )
        else:
            # Layer-wise pruning
            for name, module in self.prune_parameters:
                if "unstructured" in self.pruning_config["method"]:
                    prune_method = getattr(prune, self.pruning_config["method"])
                    prune_method(module, name="weight", amount=current_amount)
                else:
                    dim = int(self.pruning_config["method"].split("_")[0][-1])
                    prune.ln_structured(
                        module, name="weight", amount=current_amount, n=dim, dim=1
                    )
        
        # Log sparsity statistics
        self._log_sparsity()
    
    def _log_sparsity(self):
        """Log sparsity statistics to TensorBoard"""
        total_params = 0
        total_pruned = 0
        
        for name, module in self.prune_parameters:
            if isinstance(module.weight, torch.nn.parameter.Parameter):
                sparsity = torch.sum(module.weight == 0).item() / module.weight.numel()
                self.log(f"pruning/sparsity/{name}", sparsity, prog_bar=False)
                total_params += module.weight.numel()
                total_pruned += torch.sum(module.weight == 0).item()
        
        global_sparsity = total_pruned / total_params if total_params > 0 else 0
        self.log("pruning/global_sparsity", global_sparsity, prog_bar=True)
    
    def on_train_epoch_start(self):
        """Apply pruning at specified intervals"""
        current_epoch = self.current_epoch
        if (current_epoch % self.pruning_config["frequency"] == 0):
            self._apply_pruning()
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optim = getattr(torch.optim, self.pruning_config["optimizer"]["name"])
        scheduler = getattr(torch.optim.lr_scheduler, self.pruning_config["scheduler"]["name"])

        optim = optim(self.parameters(), **self.pruning_config["optimizer"]["params"])
        scheduler = scheduler(optim, **{key: eval(val) if "lambda" in key else val for key, val in self.pruning_config["scheduler"]["params"].items()})

        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
