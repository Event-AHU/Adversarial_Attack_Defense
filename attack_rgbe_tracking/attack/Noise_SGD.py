import os
import os.path as osp
import argparse  
import sys
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import Resize
from torch.optim.optimizer import Optimizer, required
import random
import numpy as np
import math
import torchvision
from torchvision import transforms

CHECK = 1e-5
SAT_MIN = 0.5


class Att_SGD(Optimizer):
    def __init__(
            self, params, lr=required, dampening=0, weight_decay=0,
            nesterov=False, max_eps=10 / 255
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Error learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Error weight_decay: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign=False,
        )
      
        super(Att_SGD, self).__init__(params, defaults)
        self.sat = 0
        self.sat_prev = 0
        self.max_eps = max_eps

    def __setstate__(self, state):
        super(Att_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)


    def rescale(self, ):
        for group in self.param_groups:
            if not group["sign"]:
                continue
            for p in group["params"]:
                self.sat_prev = self.sat
                self.sat = (p.data.abs() >= self.max_eps).sum().item() / p.data.numel()
                sat_change = abs(self.sat - self.sat_prev)
                if rescale_check(CHECK, self.sat, sat_change, SAT_MIN):
                    print('rescaled')
                    p.data = p.data / 2

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        target_threshold = 6.6e-5

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # d_p = where((d_p > target_threshold ) | (d_p < -target_threshold), d_p, 0)

                # if group["sign"]:
                #     d_p = d_p / (d_p.norm(1) + 1e-12)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
               
                if group["sign"]:
                    p.data.add_(-group["lr"] * d_p.sign())
                    # p.data = torch.clamp(p.data, -self.max_eps, self.max_eps)
                else:
                    # p.data.add_(-group["lr"], d_p)
                    p.data.add_(-group["lr"] * d_p.sign())

        return loss

def rescale_check(check, sat, sat_change, sat_min):
    return sat_change < check and sat > sat_min

def keepGradUpdate(noiseData, optimizer, gradInfo, max_eps):
    weight_decay = optimizer.param_groups[0]["weight_decay"]
    momentum = optimizer.param_groups[0]["momentum"]
    dampening = optimizer.param_groups[0]["dampening"]
    nesterov = optimizer.param_groups[0]["nesterov"]
    lr = optimizer.param_groups[0]["lr"]

    d_p = gradInfo
    # if optimizer.param_groups[0]["sign"]:
    #     d_p = d_p / (d_p.norm(1) + 1e-12)
    if weight_decay != 0:
        d_p.add_(weight_decay, noiseData)
            
    noiseData = noiseData - lr * d_p.sign()
    
    # if momentum != 0:
    #     param_state = optimizer.state[noiseData]
    #     if "momentum_buffer" not in param_state:
    #         buf = param_state["momentum_buffer"] = torch.zeros_like(noiseData.data)
    #         # buf.mul_(momentum).add_(d_p)
    #         buf = buf * momentum + d_p
    #     else:
    #         buf = param_state["momentum_buffer"]
    #         buf = buf * momentum + (1 - dampening) * d_p
    #     if nesterov:
    #         d_p = d_p + momentum * buf
    #     else:
    #         d_p = buf

        # if optimizer.param_groups[0]["sign"]:
        #     noiseData = noiseData - lr * d_p.sign()
        #     # noiseData = torch.clamp(noiseData, -max_eps, max_eps)
        # else:
        #     noiseData = noiseData - lr * d_p.sign()


    return noiseData


def where(cond, x, y):
        """
        code from :
            https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
        """ #当 cond 为 True（即 1.0）时，选择 x
        cond = cond.float()
        return (cond*x) + ((1-cond)*y)