import torch
import torch.nn.functional as F


class DistillationLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, targets):
        ce_loss = F.cross_entropy(student_logits, targets)
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits.detach() / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        return (1.0 - self.alpha) * ce_loss + self.alpha * distill_loss
