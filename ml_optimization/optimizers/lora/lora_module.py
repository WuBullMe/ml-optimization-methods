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
        
        # Freeze original parameters
        for param in original_layer.parameters():
            param.requires_grad = False
            
        # Add low-rank parameters
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.A = torch.nn.Parameter(torch.randn(in_features, rank))
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
        model_logits = torch.stack([x[0] for x in self.validation_data])[0].cpu()
        model_targets = torch.stack([x[1] for x in self.validation_data])[0].cpu()

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