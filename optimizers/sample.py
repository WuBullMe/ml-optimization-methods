import torch
from torch import nn
import pytorch_lightning as pl

# Knowledge Distillation Lightning Module
class KDLightningModule(pl.LightningModule):
    def __init__(self, student, teacher, alpha, temperature, lr):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.lr = lr

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
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student.parameters(), lr=self.lr)

# KD Optimization Class
class KDOptimization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def fit(self, teachet_model, student_model, data):
        kd_module = KDLightningModule(
            student=student_model,
            teacher=teachet_model,
            alpha=self.config.get('alpha', 0.5),
            temperature=self.config.get('temperature', 5.0),
            lr=self.config.get('lr', 0.001)
        )

        trainer = pl.Trainer(
            max_epochs=self.config.get('epochs', 10),
            accelerator='auto',
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True
        )

        trainer.fit(kd_module, data)

        return student_model