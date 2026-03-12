import copy
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


class EMATeacher:
    def __init__(self, model: AutoModelForCausalLM, alpha: float = 0.01, device: Optional[torch.device] = None):
        self.alpha = alpha
        self.model = copy.deepcopy(model)

        self.model.gradient_checkpointing_disable()
        self.model.requires_grad_(False)
        self.model.eval()

        if device is not None:
            self.model.to(device)

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def update(self, model: AutoModelForCausalLM):
        student_params = dict(model.named_parameters())

        for name, param in self.model.named_parameters():
            if name in student_params:
                student_param = student_params[name]
                param.mul_(1 - self.alpha).add_(student_param.data,
                                                alpha=self.alpha)

    @torch.no_grad()
    def sync_across_processes(self, accelerator: Accelerator):
        if accelerator.num_processes > 1:
            for param in self.model.parameters():
                torch.distributed.all_reduce(
                    param.data, op=torch.distributed.ReduceOp.AVG)
