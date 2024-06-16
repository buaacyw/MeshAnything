# -*- coding: utf-8 -*-

import torch
import numpy as np


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    # dataset = worker_info.dataset
    # split_size = dataset.num_records // worker_info.num_workers
    # # reset num_records to the true number to retain reliable length information
    # dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
    # current_id = np.random.choice(len(np.random.get_state()[1]), 1)
    # return np.random.seed(np.random.get_state()[1][current_id] + worker_id)

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


def collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """

    Args:
        samples (list[dict]):
        combine_tensors:
        combine_scalars:

    Returns:

    """

    result = {}

    keys = samples[0].keys()

    for key in keys:
        result[key] = []

    for sample in samples:
        for key in keys:
            val = sample[key]
            result[key].append(val)

    for key in keys:
        val_list = result[key]
        if isinstance(val_list[0], (int, float)):
            if combine_scalars:
                result[key] = np.array(result[key])

        elif isinstance(val_list[0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(val_list)

        elif isinstance(val_list[0], np.ndarray):
            if combine_tensors:
                result[key] = np.stack(val_list)

    return result
