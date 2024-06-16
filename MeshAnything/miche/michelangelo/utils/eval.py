# -*- coding: utf-8 -*-

import torch


def compute_psnr(x, y, data_range: float = 2, eps: float = 1e-7):

    mse = torch.mean((x - y) ** 2)
    psnr = 10 * torch.log10(data_range / (mse + eps))

    return psnr

