import copy
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score

class QuantizationModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        quantization_method: str = "qat",
        quantization_params: Dict[str, str] = None,
        bits: int = 8,
        layers: List[str] = None,
        optimizer: Dict[str, str] = None,
        scheduler: Dict[str, str] = None,
        loss: str = "cross_entropy"
    ):
        """
        PyTorch Lightning module for model quantize.
        
        Args:
            model: The model to quantize
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = copy.deepcopy(model)
        self.validation_data = []
        self.quantization_config = {
            "method": quantization_method,
            "method_params": quantization_params,
            "bits": bits,
            "layers": layers,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss": loss
        }

        if self.quantization_config['method'] in ["qat", "pqt_static"]:
            self.model = nn.Sequential(
                torch.ao.quantization.QuantStub(),
                self.model,
                torch.ao.quantization.DeQuantStub()
            )

        self._apply_quantization()
            
    def _apply_quantization(self) -> nn.Module:
        """Apply specific quantization method to model"""
        quant_type = self.quantization_config['method']
        bits = self.quantization_config.get('bits', 8)
        
        if quant_type.startswith("pqt_dynamic"):
            self._apply_dynamic_quantization()
        elif quant_type.startswith('pqt_static'):
            self._apply_static_quantization()
        elif quant_type.startswith("qat"):
            self._apply_qat()
        # elif quant_type.startswith("fp16"):
        #     return self.model.half()
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")
    
    def _apply_dynamic_quantization(self) -> nn.Module:
        """Apply dynamic quantization"""
        self.model = torch.ao.quantization.quantize_dynamic(
            self.model,
            set([getattr(nn, layer) for layer in self.quantization_config["layers"]]),
            dtype={
                8: torch.qint8,
                16: torch.float16
            }[self.quantization_config.get('bits', 8)]
        )

        self.optimize_parameters = self.model.parameters()
    
    def _apply_static_quantization(self) -> nn.Module:
        """Apply static quantization with calibration"""
        # Prepare model
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

        self.optimize_parameters = self.model.parameters()
        
        # Prepare for QAT
        torch.ao.quantization.prepare(self.model, inplace=True)
        print(self.model)

    # def _calibrate_model(self, model: nn.Module, steps: int):
    #     """Run calibration for static quantization"""
    #     # Implement calibration with sample data
    #     pass
    
    def _apply_qat(self) -> nn.Module:
        """Apply Quantization-Aware Training"""
        # Prepare model for QAT
        # model.eval()
        # model.fuse_model(is_qat=True)
        self.model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

        self.optimize_parameters = self.model.parameters()
        
        # Prepare for QAT
        torch.ao.quantization.prepare_qat(self.model, inplace=True)
        # print(self.model)
        # # Fine-tune with quantization aware training
        # optimizer = self._create_optimizer(prepared_model)
        # self._train_model(prepared_model, optimizer, self.quantization_config.get('qat_epochs', 5))
        
        # # Convert to quantized model
        # quantized_model = torch.quantization.convert(prepared_model.eval())
        # return quantized_model
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optim = getattr(torch.optim, self.quantization_config["optimizer"]["name"])
        optim = optim(self.optimize_parameters, **self.quantization_config["optimizer"]["params"])

        if self.quantization_config["scheduler"] is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.quantization_config["scheduler"]["name"])
            scheduler = scheduler(optim, **{key: eval(val) if "lambda" in key else val for key, val in self.quantization_config["scheduler"]["params"].items()})

            return [optim], [scheduler]

        return optim

    def training_step(self, batch, batch_idx):
        if self.quantization_config['method'] in ["qat", "pqt_static"]:
            if self.current_epoch > 3:
                # Freeze quantizer parameters
                self.model.apply(torch.ao.quantization.disable_observer)
            if self.current_epoch > 2:
                # Freeze batch norm mean and variance estimates
                self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        if self.quantization_config['method'] == "dynamic":
            return None

        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Convert to quantized model to evaluate
        if self.quantization_config['method'] == "qat":
            self.model.to("cpu")
            quantized_model = torch.ao.quantization.convert(self.model.eval(), inplace=False)
            quantized_model.eval()
            self.model.to("cuda")
            # print(quantized_model)
            x, y = batch
            x, y = x.to("cpu"), y.to("cpu")
            y_hat = quantized_model(x)
        else:
            if self.quantization_config['method'].startswith("pqt_static"):
                self.model.to("cpu")
                quantized_model = torch.ao.quantization.convert(self.model.eval(), inplace=False)
                quantized_model.eval()
                self.model.to("cuda")
                # print(quantized_model)
                x, y = batch
                x, y = x.to("cpu"), y.to("cpu")
                y_hat = quantized_model(x)
            else:
                x, y = batch
                y_hat = self(x)
        
        self.validation_data.append((torch.argmax(y_hat, -1), y))
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
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
