import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import recall_score, precision_score, f1_score

# Knowledge Distillation Lightning Module
class KDModule(pl.LightningModule):
    def __init__(self, student, teacher, alpha_loss, temperature, logit_std=False, optimizer=None, scheduler=None):
        super().__init__()
        self.save_hyperparameters()

        self.student = student
        self.teacher = teacher
        self.alpha_loss = alpha_loss
        self.temperature = temperature
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logit_std = logit_std
        self.validation_data = []

        # Freeze teacher parameters
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)
    
    def _standardize_logits(self, logits):
        if not self.logit_std:
            return logits
            
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True)
        standardized_logits = (logits - mean) / (std + 1e-6)
        return standardized_logits

    def _get_loss(self, student_logits, teacher_logits, y, return_all=False):
        if self.logit_std:
            student_logits = self._standardize_logits(student_logits)
            teacher_logits = self._standardize_logits(teacher_logits)

        loss_ce = nn.functional.cross_entropy(student_logits, y)
        loss_kd = nn.functional.kl_div(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        loss = self.alpha_loss * loss_ce  + (1 - self.alpha_loss) * loss_kd
        
        if return_all:
            return loss_ce, loss_kd, loss
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        # Student predictions
        student_logits = self.student(x)

        loss = self._get_loss(student_logits, teacher_logits, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        student_logits = self.student(x)
        teacher_logits = self.teacher(x)

        self.validation_data.append((torch.argmax(student_logits, -1), y))

        # Calculate losses
        loss_ce, loss_kd, loss = self._get_loss(student_logits, teacher_logits, y, return_all=True)

        losses = {
            'val_loss': loss,
            'val_student_ce_loss': loss_ce,
            'val_kl_div_loss': loss_kd,
        }

        self.log_dict(losses)
        return losses
    
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


    def configure_optimizers(self):
        optim = getattr(torch.optim, self.optimizer["name"])
        optim = optim(self.student.parameters(), **self.optimizer["params"])

        if self.scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler["name"])
            scheduler = scheduler(optim, **{key: eval(val) if "lambda" in key else val for key, val in self.scheduler["params"].items()})

            return [optim], [scheduler]
        
        return optim