import copy
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn

class QuantizationModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        quantization_method: str = "fp16",
        quantization_params: Dict[str, str] = None,
        bits: int = 8,
        layers: List[str] = None,
        optimizer: Dict[str, str] = None,
        scheduler: Dict[str, str] = None
    ):
        """
        PyTorch Lightning module for model pruning.
        
        Args:
            model: The model to prune
        """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.quantization_config = {
            "method": quantization_method,
            "method_params": quantization_params,
            "bits": bits,
            "layers": layers,
            "optimizer": optimizer,
            "scheduler": scheduler
        }

        self._apply_quantization()
            
    def _apply_quantization(self) -> nn.Module:
        """Apply specific quantization method to model"""
        quant_type = self.quantization_config['method']
        bits = self.quantization_config.get('bits', 8)
        
        if quant_type.startswith("dynamic"):
            return self._apply_dynamic_quantization(self.model)
        # elif quant_type == 'static':
        #     return self._apply_static_quantization(self.model)
        elif quant_type.startswith("qat"):
            return self._apply_qat(self.model)
        elif quant_type.startswith("fp16"):
            return self.model.half()
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")
    
    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization"""
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {
                nn.Linear,
                nn.Conv2d
            },
            dtype={
                8: torch.qint8,
                16: torch.float16
            }[self.quantization_config.get('bits', 8)]
        )
        return quantized_model
    
    # def _apply_static_quantization(self, model: nn.Module) -> nn.Module:
    #     """Apply static quantization with calibration"""
    #     # Prepare model
    #     model.eval()
    #     model.fuse_model()
        
    #     # Set quantization config
    #     model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
    #     # Prepare for calibration
    #     prepared_model = torch.quantization.prepare(model)
        
    #     # Calibrate with sample data
    #     self._calibrate_model(prepared_model, self.quantization_config.get('calibration_steps', 100))
        
    #     # Convert to quantized model
    #     quantized_model = torch.quantization.convert(prepared_model)
    #     return quantized_model

    # def _calibrate_model(self, model: nn.Module, steps: int):
    #     """Run calibration for static quantization"""
    #     # Implement calibration with sample data
    #     pass
    
    def _apply_qat(self, model: nn.Module, method: Dict) -> nn.Module:
        """Apply Quantization-Aware Training"""
        # Prepare model for QAT
        model.train()
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        prepared_model = torch.quantization.prepare_qat(model)
        
        # Fine-tune with quantization aware training
        optimizer = self._create_optimizer(prepared_model)
        self._train_model(prepared_model, optimizer, method.get('qat_epochs', 5))
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model.eval())
        return quantized_model
        
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
