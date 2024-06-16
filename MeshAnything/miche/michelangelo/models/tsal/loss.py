# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Dict

from MeshAnything.miche.michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from MeshAnything.miche.michelangelo.utils.eval import compute_psnr
from MeshAnything.miche.michelangelo.utils import misc


class KLNearFar(nn.Module):
    def __init__(self,
                 near_weight: float = 0.1,
                 kl_weight: float = 1.0,
                 num_near_samples: Optional[int] = None):

        super().__init__()

        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.num_near_samples = num_near_samples
        self.geo_criterion = nn.BCEWithLogitsLoss()

    def forward(self,
                posteriors: Optional[DiagonalGaussianDistribution],
                logits: torch.FloatTensor,
                labels: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        """

        Args:
            posteriors (DiagonalGaussianDistribution or torch.distributions.Normal):
            logits (torch.FloatTensor): [B, 2*N], logits[:, 0:N] is the volume points; logits[:, N:2N] is the near points;
            labels (torch.FloatTensor): [B, 2*N], labels[:, 0:N] is the volume points; labels[:, N:2N] is the near points;
            split (str):
            **kwargs:

        Returns:
            loss (torch.Tensor): (,)
            log (dict):

        """

        if self.num_near_samples is None:
            num_vol = logits.shape[1] // 2
        else:
            num_vol = logits.shape[1] - self.num_near_samples

        vol_logits = logits[:, 0:num_vol]
        vol_labels = labels[:, 0:num_vol]

        near_logits = logits[:, num_vol:]
        near_labels = labels[:, num_vol:]

        # occupancy loss
        # vol_bce = self.geo_criterion(vol_logits, vol_labels)
        # near_bce = self.geo_criterion(near_logits, near_labels)
        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=vol_logits.dtype, device=vol_logits.device)
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        loss = vol_bce + near_bce * self.near_weight + kl_loss * self.kl_weight

        with torch.no_grad():
            preds = logits >= 0
            accuracy = (preds == labels).float()
            accuracy = accuracy.mean()
            pos_ratio = torch.mean(labels)

        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/near".format(split): near_bce.detach(),
            "{}/far".format(split): vol_bce.detach(),
            "{}/kl".format(split): kl_loss.detach(),
            "{}/accuracy".format(split): accuracy,
            "{}/pos_ratio".format(split): pos_ratio
        }

        if posteriors is not None:
            log[f"{split}/mean"] = posteriors.mean.mean().detach()
            log[f"{split}/std_mean"] = posteriors.std.mean().detach()
            log[f"{split}/std_max"] = posteriors.std.max().detach()

        return loss, log


class KLNearFarColor(nn.Module):
    def __init__(self,
                 near_weight: float = 0.1,
                 kl_weight: float = 1.0,
                 color_weight: float = 1.0,
                 color_criterion: str = "mse",
                 num_near_samples: Optional[int] = None):

        super().__init__()

        self.color_weight = color_weight
        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.num_near_samples = num_near_samples

        if color_criterion == "mse":
            self.color_criterion = nn.MSELoss()

        elif color_criterion == "l1":
            self.color_criterion = nn.L1Loss()

        else:
            raise ValueError(f"{color_criterion} must be [`mse`, `l1`].")

        self.geo_criterion = nn.BCEWithLogitsLoss()

    def forward(self,
                posteriors: Optional[DiagonalGaussianDistribution],
                logits: torch.FloatTensor,
                labels: torch.FloatTensor,
                pred_colors: torch.FloatTensor,
                gt_colors: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        """

        Args:
            posteriors (DiagonalGaussianDistribution or torch.distributions.Normal):
            logits (torch.FloatTensor): [B, 2*N], logits[:, 0:N] is the volume points; logits[:, N:2N] is the near points;
            labels (torch.FloatTensor): [B, 2*N], labels[:, 0:N] is the volume points; labels[:, N:2N] is the near points;
            pred_colors (torch.FloatTensor): [B, M, 3]
            gt_colors (torch.FloatTensor): [B, M, 3]
            split (str):
            **kwargs:

        Returns:
            loss (torch.Tensor): (,)
            log (dict):

        """

        if self.num_near_samples is None:
            num_vol = logits.shape[1] // 2
        else:
            num_vol = logits.shape[1] - self.num_near_samples

        vol_logits = logits[:, 0:num_vol]
        vol_labels = labels[:, 0:num_vol]

        near_logits = logits[:, num_vol:]
        near_labels = labels[:, num_vol:]

        # occupancy loss
        # vol_bce = self.geo_criterion(vol_logits, vol_labels)
        # near_bce = self.geo_criterion(near_logits, near_labels)
        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        # surface color loss
        color = self.color_criterion(pred_colors, gt_colors)

        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=pred_colors.dtype, device=pred_colors.device)
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        loss = vol_bce + near_bce * self.near_weight + color * self.color_weight + kl_loss * self.kl_weight

        with torch.no_grad():
            preds = logits >= 0
            accuracy = (preds == labels).float()
            accuracy = accuracy.mean()
            psnr = compute_psnr(pred_colors, gt_colors)

        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/near".format(split): near_bce.detach(),
            "{}/far".format(split): vol_bce.detach(),
            "{}/color".format(split): color.detach(),
            "{}/kl".format(split): kl_loss.detach(),
            "{}/psnr".format(split): psnr.detach(),
            "{}/accuracy".format(split): accuracy
        }

        return loss, log


class ContrastKLNearFar(nn.Module):
    def __init__(self,
                 contrast_weight: float = 1.0,
                 near_weight: float = 0.1,
                 kl_weight: float = 1.0,
                 num_near_samples: Optional[int] = None):

        super().__init__()

        self.labels = None
        self.last_local_batch_size = None

        self.contrast_weight = contrast_weight
        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.num_near_samples = num_near_samples
        self.geo_criterion = nn.BCEWithLogitsLoss()

    def forward(self,
                shape_embed: torch.FloatTensor,
                text_embed: torch.FloatTensor,
                image_embed: torch.FloatTensor,
                logit_scale: torch.FloatTensor,
                posteriors: Optional[DiagonalGaussianDistribution],
                shape_logits: torch.FloatTensor,
                shape_labels: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs):

        local_batch_size = shape_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * misc.get_rank() + torch.arange(
                local_batch_size, device=shape_embed.device
            ).long()
            self.last_local_batch_size = local_batch_size

        # normalized features
        shape_embed = F.normalize(shape_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        shape_embed_all, text_embed_all, image_embed_all = misc.all_gather_batch(
            [shape_embed, text_embed, image_embed]
        )

        # cosine similarity as logits
        logits_per_shape_text = logit_scale * shape_embed @ text_embed_all.t()
        logits_per_text_shape = logit_scale * text_embed @ shape_embed_all.t()
        logits_per_shape_image = logit_scale * shape_embed @ image_embed_all.t()
        logits_per_image_shape = logit_scale * image_embed @ shape_embed_all.t()
        contrast_loss = (F.cross_entropy(logits_per_shape_text, self.labels) +
                         F.cross_entropy(logits_per_text_shape, self.labels)) / 2 + \
                        (F.cross_entropy(logits_per_shape_image, self.labels) +
                         F.cross_entropy(logits_per_image_shape, self.labels)) / 2

        # shape reconstruction
        if self.num_near_samples is None:
            num_vol = shape_logits.shape[1] // 2
        else:
            num_vol = shape_logits.shape[1] - self.num_near_samples

        vol_logits = shape_logits[:, 0:num_vol]
        vol_labels = shape_labels[:, 0:num_vol]

        near_logits = shape_logits[:, num_vol:]
        near_labels = shape_labels[:, num_vol:]

        # occupancy loss
        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=vol_logits.dtype, device=vol_logits.device)
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        loss = vol_bce + near_bce * self.near_weight + kl_loss * self.kl_weight + contrast_loss * self.contrast_weight

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_shape_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            shape_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_shape_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            shape_image_acc = 100 * correct / local_batch_size

            preds = shape_logits >= 0
            accuracy = (preds == shape_labels).float()
            accuracy = accuracy.mean()

            log = {
                "{}/contrast".format(split): contrast_loss.clone().detach(),
                "{}/near".format(split): near_bce.detach(),
                "{}/far".format(split): vol_bce.detach(),
                "{}/kl".format(split): kl_loss.detach(),
                "{}/shape_text_acc".format(split): shape_text_acc,
                "{}/shape_image_acc".format(split): shape_image_acc,
                "{}/total_loss".format(split): loss.clone().detach(),
                "{}/accuracy".format(split): accuracy,
            }

            if posteriors is not None:
                log[f"{split}/mean"] = posteriors.mean.mean().detach()
                log[f"{split}/std_mean"] = posteriors.std.mean().detach()
                log[f"{split}/std_max"] = posteriors.std.max().detach()

        return loss, log
