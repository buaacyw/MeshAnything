# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseDenoiser(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, t, context):
        raise NotImplementedError
