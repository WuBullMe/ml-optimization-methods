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