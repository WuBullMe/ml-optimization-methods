import torch
import copy
import pytorch_lightning as pl

from .lora_module import LoRALayer, LoRAModule

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