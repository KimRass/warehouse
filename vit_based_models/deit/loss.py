# References
    # https://github.com/facebookresearch/deit/blob/main/losses.py

import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(
        self,
        cls_criterion: torch.nn.Module,
        teacher: torch.nn.Module,
        distil_type: str,
        lamb: float,
        tau: float
    ):
        super().__init__()

        assert distil_type in ["none", "soft", "hard"]

        self.cls_criterion = cls_criterion # Usually cross entropy loss
        self.teacher = teacher
        self.distil_type = distil_type
        self.lamb = lamb
        self.tau = tau

        # `cls_logit`, `distil_logit`: $Z_{s}$
    def forward(self, input, cls_logit, distil_logit, label):
        cls_loss = self.cls_criterion(cls_logit, label) # $\mathcal{L}_{\text{CE}}$
        # cls_loss = self.cls_criterion(F.softmax(cls_logit, dim=1), label) # $\mathcal{L}_{\text{CE}}$
        if self.distil_type == "none":
            return cls_loss

        # Do not backpropogate through the teacher model.
        with torch.no_grad():
            teacher_logit = self.teacher(input) # $Z_{t}$

        if self.distil_type == "soft":
            # We provide the teacher"s targets in log probability because we use log_target=True 
            # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
            # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
            # We divide by distil_logit.numel() to have the legacy PyTorch behavior. 
            # But we also experiments output_kd.size(0) 
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
            distil_loss =  F.kl_div(
                # F.log_softmax(distil_logit / self.tau, dim=1),
                # F.log_softmax(teacher_logit / self.tau, dim=1),
                F.softmax(distil_logit / self.tau, dim=1),
                F.softmax(teacher_logit / self.tau, dim=1),
                reduction="sum",
                # log_target=True
                log_target=False
            )
            distil_loss *= self.tau ** 2
            distil_loss /= distil_logit.numel()

        elif self.distil_type == "hard":
            hard_decision = teacher_logit.argmax(dim=1) # $y_{t}$
            distil_loss = F.cross_entropy(distil_logit, hard_decision)
            # distil_loss = F.cross_entropy(F.softmax(distil_logit, dim=1), hard_decision)

        loss = (1 - self.lamb) * cls_loss + self.lamb * distil_loss
        return loss
