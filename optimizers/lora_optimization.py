import torch
import torch.nn as nn
import copy
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


# LoRA Optimization Class
class LoROptimization(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        self._setup_config_training()
    
    def _setup_config_training(self):
        def get_callback(cb_config):
            cls = getattr(pl.callbacks, cb_config["class"])
            return cls(**cb_config.get("params", {}))

        def get_logger(log_config):
            return pl.loggers.TensorBoardLogger(**log_config.get("params", {}))
        
        self.setup_training = {
            "callbacks": [get_callback(cb) for cb in self.config["callbacks"]],
            "logger": get_logger(self.config["logging"]),
            **self.config["hardware"],
            **self.config["train"]
        }

    def _apply_lora(self, module, rank):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                # Replace Linear layer with LoRALayer
                setattr(module, name, LoRALayer(child, rank))
            else:
                # Recursively apply to child modules
                self._apply_lora(child, rank)

    def fit(self, model, dataloaders):
        # Create a copy of the model
        model = copy.deepcopy(model)
        
        # Apply LoRA to all Linear layers
        self._apply_lora(model, self.config['rank'])
        pl_model = LoRAModule(model, **self.config["LoRAModule"])
        
        trainer = pl.Trainer(
            **self.setup_training
        )

        trainer.fit(pl_model, dataloaders["train"], dataloaders["val"])

        return model