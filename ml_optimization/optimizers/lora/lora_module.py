import torch
import torch.nn as nn
import pytorch_lightning as pl

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
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss = nn.CrossEntropyLoss()
    
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

        self.log('val_loss', loss)
        return loss

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