from typing import Tuple

import numpy as np
import torch
import torch
from torch import nn

def init_alpha_target(true_values,target_values,device): #true是真p target是null_p
    
    mask = torch.arange(3).unsqueeze(-1) # 0 1 2
    # 真的p
    true_alpha = ((true_values + 1) == mask).float().to(device).T
        
    target_alpha = ((target_values + 1) == mask).float().to(device).T
    target_alpha[:, 0] = 0.05
    target_alpha[:, 1] = 0.9
    target_alpha[:, 2] = 0.05

    combine_alpha = torch.cat([true_alpha, target_alpha], dim=0).to(device)

    return combine_alpha


alpha = init_alpha_target()

alpha = nn.parameter.Parameter(data=alpha, requires_grad=True)

#c初始化
hard_values = torch.argmax(alpha, dim=-1).unsqueeze(-1) - 1

hard_true_value, true_value_for_attack = hard_values[0].squeeze(-1), hard_values.squeeze(-1)


def forward(
        self, alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        alpha = alpha.unsqueeze(0).repeat_interleave(
            self.sample_num, dim=0
        )  # [sample_number, event_number, 3]
        soften_gumbel_3d = gumbel_softmax(alpha, tau=self.tau, hard=False)
        hard_event: torch.Tensor = self.hard_argmax.apply(
            soften_gumbel_3d
        )  # type:ignore
        if self.use_soft_event:
            soft_event = self.soft_argmax(soften_gumbel_3d)
            hard_event.detach_()
        else:
            soft_event = None
        return hard_event, soft_event  # type:ignore hard_event [sample_number, event_number]
