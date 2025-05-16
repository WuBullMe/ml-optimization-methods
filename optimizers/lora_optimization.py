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
    
class LoRALightningModule(pl.LightningModule):
    def __init__(self, model, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.config = config
        self.loss = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        outputs = self.model(x)
        loss = self.loss(outputs, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Setup optimizer (only train LoRA parameters)
        lora_params = []
        for m in self.model.modules():
            if isinstance(m, LoRALayer):
                lora_params += [m.A, m.B]

        return torch.optim.Adam(lora_params, lr=self.config['optimization']['lr'])


# LoRA Optimization Class
class LoROptimization(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def _apply_lora(self, module, rank):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                # Replace Linear layer with LoRALayer
                setattr(module, name, LoRALayer(child, rank))
            else:
                # Recursively apply to child modules
                self._apply_lora(child, rank)

    def fit(self, model, data, config):
        # Create a copy of the model
        model = copy.deepcopy(model)
        
        # Apply LoRA to all Linear layers
        self._apply_lora(model, self.config['rank'])
        pl_model = LoRALightningModule(model, config)
        
        trainer = pl.Trainer(
            max_epochs=self.config.get('epochs', 10),
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            max_steps=1,
            num_sanity_val_steps=1
        )

        trainer.fit(pl_model, data)

        return model