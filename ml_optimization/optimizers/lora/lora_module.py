import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import recall_score, precision_score, f1_score

# LoRA Layer Implementation
class LoRALayer(torch.nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank

        # Add low-rank parameters
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.A = torch.nn.Parameter(torch.zeros(in_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.B = torch.nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = x @ self.A @ self.B
        return original_output + lora_output


class LoRAModule(pl.LightningModule):
    def __init__(self, model, optimizer=None, scheduler=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.validation_data = []

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        outputs = self.model(x)
        loss = self.loss(outputs, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        outputs = self.model(x)
        loss = self.loss(outputs, y)

        self.validation_data.append((torch.argmax(outputs, -1), y))

        self.log('val_loss', loss)
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

    def configure_optimizers(self):
        # Setup optimizer (only train LoRA parameters)
        lora_params = []
        for m in self.model.modules():
            if isinstance(m, LoRALayer):
                lora_params += [m.A, m.B]

        optim = getattr(torch.optim, self.optimizer["name"])
        optim = optim(lora_params, **self.optimizer["params"])

        if self.scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler["name"])
            scheduler = scheduler(optim, **{key: eval(val) if "lambda" in key else val for key, val in self.scheduler["params"].items()})

            return [optim], [scheduler]

        return optim