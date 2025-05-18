import torch
from torch import nn
import pytorch_lightning as pl

# Knowledge Distillation Lightning Module
class KDModule(pl.LightningModule):
    def __init__(self, student, teacher, alpha_loss, temperature, optimizer=None, scheduler=None):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha_loss = alpha_loss
        self.temperature = temperature
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Student predictions
        student_logits = self.student(x)

        # Calculate losses
        loss_ce = nn.functional.cross_entropy(student_logits, y)
        loss_kd = nn.functional.kl_div(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha_loss * loss_ce + (1 - self.alpha_loss) * loss_kd

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        student_logits = self.student(x)
        teacher_logits = self.teacher(x)

        # Calculate losses
        loss_ce = nn.functional.cross_entropy(student_logits, y)
        loss_kd = nn.functional.kl_div(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha_loss * loss_ce + (1 - self.alpha_loss) * loss_kd

        losses = {
            'val_loss': loss,
            'val_student_ce_loss': loss_ce,
            'val_kl_div_loss': loss_kd,
        }

        self.log_dict(losses)
        return losses

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.optimizer["name"])
        optim = optim(self.student.parameters(), **self.optimizer["params"])

        if self.scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler["name"])
            scheduler = scheduler(optim, **{key: eval(val) if "lambda" in key else val for key, val in self.scheduler["params"].items()})

            return [optim], [scheduler]
        
        return optim

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