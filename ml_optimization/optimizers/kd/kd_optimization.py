from torch import nn
import pytorch_lightning as pl

from .kd_module import KDModule

# KD Optimization Class
class KDOptimization(nn.Module):
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

    def fit(self, teacher_model, student_model, dataloaders):
        kd_module = KDModule(
            student=student_model,
            teacher=teacher_model,
            **self.config["KDModule"]
        )

        trainer = pl.Trainer(
            **self.setup_training
        )

        trainer.fit(kd_module, dataloaders["train"], dataloaders["val"])

        return student_model