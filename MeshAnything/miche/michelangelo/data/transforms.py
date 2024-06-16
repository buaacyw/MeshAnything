# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import warnings
import random
from omegaconf.listconfig import ListConfig
from webdataset import pipelinefilter
import torch
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode
from torchvision.transforms.transforms import _interpolation_modes_from_int
from typing import Sequence

from MeshAnything.miche.michelangelo.utils import instantiate_from_config


def _uid_buffer_pick(buf_dict, rng):
    uid_keys = list(buf_dict.keys())
    selected_uid = rng.choice(uid_keys)
    buf = buf_dict[selected_uid]

    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()

    if len(buf) == 0:
        del buf_dict[selected_uid]

    return sample


def _add_to_buf_dict(buf_dict, sample):
    key = sample["__key__"]
    uid, uid_sample_id = key.split("_")
    if uid not in buf_dict:
        buf_dict[uid] = []
    buf_dict[uid].append(sample)

    return buf_dict


def _uid_shuffle(data, bufsize=1000, initial=100, rng=None, handler=None):
    """Shuffle the data in the stream.

    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator
    rng: either random module or random.Random instance

    """
    if rng is None:
        rng = random.Random(int((os.getpid() + time.time()) * 1e9))
    initial = min(initial, bufsize)
    buf_dict = dict()
    current_samples = 0
    for sample in data:
        _add_to_buf_dict(buf_dict, sample)
        current_samples += 1

        if current_samples < bufsize:
            try:
                _add_to_buf_dict(buf_dict, next(data))  # skipcq: PYL-R1708
                current_samples += 1
            except StopIteration:
                pass

        if current_samples >= initial:
            current_samples -= 1
            yield _uid_buffer_pick(buf_dict, rng)

    while current_samples > 0:
        current_samples -= 1
        yield _uid_buffer_pick(buf_dict, rng)


uid_shuffle = pipelinefilter(_uid_shuffle)


class RandomSample(object):
    def __init__(self,
                 num_volume_samples: int = 1024,
                 num_near_samples: int = 1024):

        super().__init__()

        self.num_volume_samples = num_volume_samples
        self.num_near_samples = num_near_samples

    def __call__(self, sample):
        rng = np.random.default_rng()

        # 1. sample surface input
        total_surface = sample["surface"]
        ind = rng.choice(total_surface.shape[0], replace=False)
        surface = total_surface[ind]

        # 2. sample volume/near geometric points
        vol_points = sample["vol_points"]
        vol_label = sample["vol_label"]
        near_points = sample["near_points"]
        near_label = sample["near_label"]

        ind = rng.choice(vol_points.shape[0], self.num_volume_samples, replace=False)
        vol_points = vol_points[ind]
        vol_label = vol_label[ind]
        vol_points_labels = np.concatenate([vol_points, vol_label[:, np.newaxis]], axis=1)

        ind = rng.choice(near_points.shape[0], self.num_near_samples, replace=False)
        near_points = near_points[ind]
        near_label = near_label[ind]
        near_points_labels = np.concatenate([near_points, near_label[:, np.newaxis]], axis=1)

        # concat sampled volume and near points
        geo_points = np.concatenate([vol_points_labels, near_points_labels], axis=0)

        sample = {
            "surface": surface,
            "geo_points": geo_points
        }

        return sample


class SplitRandomSample(object):
    def __init__(self,
                 use_surface_sample: bool = False,
                 num_surface_samples: int = 4096,
                 num_volume_samples: int = 1024,
                 num_near_samples: int = 1024):

        super().__init__()

        self.use_surface_sample = use_surface_sample
        self.num_surface_samples = num_surface_samples
        self.num_volume_samples = num_volume_samples
        self.num_near_samples = num_near_samples

    def __call__(self, sample):

        rng = np.random.default_rng()

        # 1. sample surface input
        surface = sample["surface"]

        if self.use_surface_sample:
            replace = surface.shape[0] < self.num_surface_samples
            ind = rng.choice(surface.shape[0], self.num_surface_samples, replace=replace)
            surface = surface[ind]

        # 2. sample volume/near geometric points
        vol_points = sample["vol_points"]
        vol_label = sample["vol_label"]
        near_points = sample["near_points"]
        near_label = sample["near_label"]

        ind = rng.choice(vol_points.shape[0], self.num_volume_samples, replace=False)
        vol_points = vol_points[ind]
        vol_label = vol_label[ind]
        vol_points_labels = np.concatenate([vol_points, vol_label[:, np.newaxis]], axis=1)

        ind = rng.choice(near_points.shape[0], self.num_near_samples, replace=False)
        near_points = near_points[ind]
        near_label = near_label[ind]
        near_points_labels = np.concatenate([near_points, near_label[:, np.newaxis]], axis=1)

        # concat sampled volume and near points
        geo_points = np.concatenate([vol_points_labels, near_points_labels], axis=0)

        sample = {
            "surface": surface,
            "geo_points": geo_points
        }

        return sample


class FeatureSelection(object):

    VALID_SURFACE_FEATURE_DIMS = {
        "none": [0, 1, 2],                              # xyz
        "watertight_normal": [0, 1, 2, 3, 4, 5],        # xyz, normal
        "normal": [0, 1, 2, 6, 7, 8]
    }

    def __init__(self, surface_feature_type: str):

        self.surface_feature_type = surface_feature_type
        self.surface_dims = self.VALID_SURFACE_FEATURE_DIMS[surface_feature_type]

    def __call__(self, sample):
        sample["surface"] = sample["surface"][:, self.surface_dims]
        return sample


class AxisScaleTransform(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True, jitter_scale=0.005):
        assert isinstance(interval, (tuple, list, ListConfig))
        self.interval = interval
        self.min_val = interval[0]
        self.max_val = interval[1]
        self.inter_size = interval[1] - interval[0]
        self.jitter = jitter
        self.jitter_scale = jitter_scale

    def __call__(self, sample):

        surface = sample["surface"][..., 0:3]
        geo_points = sample["geo_points"][..., 0:3]

        scaling = torch.rand(1, 3) * self.inter_size + self.min_val
        # print(scaling)
        surface = surface * scaling
        geo_points = geo_points * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        geo_points *= scale

        if self.jitter:
            surface += self.jitter_scale * torch.randn_like(surface)
            surface.clamp_(min=-1.015, max=1.015)

        sample["surface"][..., 0:3] = surface
        sample["geo_points"][..., 0:3] = geo_points

        return sample


class ToTensor(object):

    def __init__(self, tensor_keys=("surface", "geo_points", "tex_points")):
        self.tensor_keys = tensor_keys

    def __call__(self, sample):
        for key in self.tensor_keys:
            if key not in sample:
                continue

            sample[key] = torch.tensor(sample[key], dtype=torch.float32)

        return sample


class AxisScale(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True, jitter_scale=0.005):
        assert isinstance(interval, (tuple, list, ListConfig))
        self.interval = interval
        self.jitter = jitter
        self.jitter_scale = jitter_scale

    def __call__(self, surface, *args):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        # print(scaling)
        surface = surface * scaling
        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale

        args_outputs = []
        for _arg in args:
            _arg = _arg * scaling * scale
            args_outputs.append(_arg)

        if self.jitter:
            surface += self.jitter_scale * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        if len(args) == 0:
            return surface
        else:
            return surface, *args_outputs


class RandomResize(torch.nn.Module):
    """Apply randomly Resize with a given probability."""

    def __init__(
        self,
        size,
        resize_radio=(0.5, 1),
        allow_resize_interpolations=(InterpolationMode.BICUBIC, InterpolationMode.BILINEAR, InterpolationMode.BILINEAR),
        interpolation=InterpolationMode.BICUBIC,
        max_size=None,
        antialias=None,
    ):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")

        self.size = size
        self.max_size = max_size
        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

        self.resize_radio = resize_radio
        self.allow_resize_interpolations = allow_resize_interpolations

    def random_resize_params(self):
        radio = torch.rand(1) * (self.resize_radio[1] - self.resize_radio[0]) + self.resize_radio[0]

        if isinstance(self.size, int):
            size = int(self.size * radio)
        elif isinstance(self.size, Sequence):
            size = list(self.size)
            size = (int(size[0] * radio), int(size[1] * radio))
        else:
            raise RuntimeError()

        interpolation = self.allow_resize_interpolations[
            torch.randint(low=0, high=len(self.allow_resize_interpolations), size=(1,))
        ]
        return size, interpolation

    def forward(self, img):
        size, interpolation = self.random_resize_params()
        img = TVF.resize(img, size, interpolation, self.max_size, self.antialias)
        img = TVF.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
        return img

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value},"
        detail += f"max_size={self.max_size}, antialias={self.antialias}), resize_radio={self.resize_radio}"
        return f"{self.__class__.__name__}{detail}"


class Compose(object):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def identity(*args, **kwargs):
    if len(args) == 1:
        return args[0]
    else:
        return args


def build_transforms(cfg):

    if cfg is None:
        return identity

    transforms = []

    for transform_name, cfg_instance in cfg.items():
        transform_instance = instantiate_from_config(cfg_instance)
        transforms.append(transform_instance)
        print(f"Build transform: {transform_instance}")

    transforms = Compose(transforms)

    return transforms

