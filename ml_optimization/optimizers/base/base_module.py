import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import recall_score, precision_score, f1_score

# Base Lightning Module
class BaseModule(pl.LightningModule):
    def __init__(self, model, optimizer=None, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.validation_data = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        model_logits = self.model(x)
        loss = nn.functional.cross_entropy(model_logits, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        model_logits = self.model(x)

        self.validation_data.append((torch.argmax(model_logits, -1), y))

        loss = nn.functional.cross_entropy(model_logits, y)
        self.log('val_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        model_logits = torch.cat([x[0] for x in self.validation_data]).cpu()
        model_targets = torch.cat([x[1] for x in self.validation_data]).cpu()

        metrics = {
            'val_accuracy': (model_logits == model_targets).float().mean().item(),
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
        self.log_dict(metrics)
        self.validation_data.clear()
        return metrics


    def configure_optimizers(self):
        optim = getattr(torch.optim, self.optimizer["name"])
        optim = optim(self.model.parameters(), **self.optimizer["params"])

        if self.scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler["name"])
            scheduler = scheduler(optim, **{key: eval(val) if "lambda" in key else val for key, val in self.scheduler["params"].items()})

            return [optim], [scheduler]
        
        return optim