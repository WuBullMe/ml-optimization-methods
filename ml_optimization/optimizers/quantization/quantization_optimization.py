import pytorch_lightning as pl
import torch.nn as nn
import torch

from .quantization_module import QuantizationModule

class QuantizationOptimization(nn.Module):
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
        
    
    def fit(self, 
            master_model, 
            dataloaders
            ):

        quantization = QuantizationModule(
            model=master_model,
            **self.config["QuantizationModule"]
        )

        trainer = pl.Trainer(
            **self.setup_training
        )

        trainer.fit(quantization, dataloaders["train"], dataloaders["val"])

        if self.config["QuantizationModule"]["quantization_method"] == "qat":
            quantized_model = torch.quantization.convert(prepared_model.eval())