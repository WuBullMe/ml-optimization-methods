import pytorch_lightning as pl
import torch.nn as nn

from .prune_module import PruningModule

class PruneOptimization(nn.Module):
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
        
        # pruner = PruningModule(
        #     model=master_model,
        #     pruning_method=self.config["prune"]["prune_type"],
        #     amount=self.config["prune"]["total_amount"] / (self.config["prune"]["max_epochs"] / self.config["prune"]["prune_every_n_epochs"]),
        #     layers_to_prune=self.config["prune"]["layers"],
        #     global_pruning=self.config["prune"]["global_pruning"],
        #     prune_every_n_epochs=self.config["prune"]["prune_every_n_epochs"],
        #     optimizer=self.config["optimizer"],
        #     scheduler=self.config["scheduler"]
        # )

        pruner = PruningModule(
            model=master_model,
            **self.config["PruningModule"]
        )

        trainer = pl.Trainer(
            **self.setup_training
        )

        trainer.fit(pruner, dataloaders["train"], dataloaders["val"])