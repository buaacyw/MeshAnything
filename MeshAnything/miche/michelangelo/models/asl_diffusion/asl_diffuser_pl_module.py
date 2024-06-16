# -*- coding: utf-8 -*-

from omegaconf import DictConfig
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from einops import rearrange

from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)

from MeshAnything.miche.michelangelo.utils import instantiate_from_config
# from MeshAnything.miche.michelangelo.models.tsal.tsal_base import ShapeAsLatentPLModule
from MeshAnything.miche.michelangelo.models.tsal.tsal_base import AlignedShapeAsLatentPLModule
from MeshAnything.miche.michelangelo.models.asl_diffusion.inference_utils import ddim_sample

SchedulerType = Union[DDIMScheduler, KarrasVeScheduler, DPMSolverMultistepScheduler]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ASLDiffuser(pl.LightningModule):
    first_stage_model: Optional[AlignedShapeAsLatentPLModule]
    # cond_stage_model: Optional[Union[nn.Module, pl.LightningModule]]
    model: nn.Module

    def __init__(self, *,
                 first_stage_config,
                 denoiser_cfg,
                 scheduler_cfg,
                 optimizer_cfg,
                 loss_cfg,
                 first_stage_key: str = "surface",
                 cond_stage_key: str = "image",
                 cond_stage_trainable: bool = True,
                 scale_by_std: bool = False,
                 z_scale_factor: float = 1.0,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.cond_stage_trainable = cond_stage_trainable

        # 1. initialize first stage. 
        # Note: the condition model contained in the first stage model.
        self.first_stage_config = first_stage_config
        self.first_stage_model = None
        # self.instantiate_first_stage(first_stage_config)

        # 2. initialize conditional stage
        # self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_model = {
            "image": self.encode_image,
            "image_unconditional_embedding": self.empty_img_cond,
            "text": self.encode_text,
            "text_unconditional_embedding": self.empty_text_cond,
            "surface": self.encode_surface,
            "surface_unconditional_embedding": self.empty_surface_cond,
        }

        # 3. diffusion model
        self.model = instantiate_from_config(
            denoiser_cfg, device=None, dtype=None
        )

        self.optimizer_cfg = optimizer_cfg

        # 4. scheduling strategy
        self.scheduler_cfg = scheduler_cfg

        self.noise_scheduler: DDPMScheduler = instantiate_from_config(scheduler_cfg.noise)
        self.denoise_scheduler: SchedulerType = instantiate_from_config(scheduler_cfg.denoise)

        # 5. loss configures
        self.loss_cfg = loss_cfg

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        self.first_stage_model = self.first_stage_model.to(self.device)

    # def instantiate_cond_stage(self, config):
    #     if not self.cond_stage_trainable:
    #         if config == "__is_first_stage__":
    #             print("Using first stage also as cond stage.")
    #             self.cond_stage_model = self.first_stage_model
    #         elif config == "__is_unconditional__":
    #             print(f"Training {self.__class__.__name__} as an unconditional model.")
    #             self.cond_stage_model = None
    #             # self.be_unconditional = True
    #         else:
    #             model = instantiate_from_config(config)
    #             self.cond_stage_model = model.eval()
    #             self.cond_stage_model.train = disabled_train
    #             for param in self.cond_stage_model.parameters():
    #                 param.requires_grad = False
    #     else:
    #         assert config != "__is_first_stage__"
    #         assert config != "__is_unconditional__"
    #         model = instantiate_from_config(config)
    #         self.cond_stage_model = model

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    @property
    def zero_rank(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return zero_rank

    def configure_optimizers(self) -> Tuple[List, List]:

        lr = self.learning_rate

        trainable_parameters = list(self.model.parameters())
        # if the conditional encoder is trainable

        # if self.cond_stage_trainable:
        #     conditioner_params = [p for p in self.cond_stage_model.parameters() if p.requires_grad]
        #     trainable_parameters += conditioner_params
        #     print(f"number of trainable conditional parameters: {len(conditioner_params)}.")

        if self.optimizer_cfg is None:
            optimizers = [torch.optim.AdamW(trainable_parameters, lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)]
            schedulers = []
        else:
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=trainable_parameters)
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            optimizers = [optimizer]
            schedulers = [scheduler]

        return optimizers, schedulers

    @torch.no_grad()
    def encode_text(self, text):

        b = text.shape[0]
        text_tokens = rearrange(text, "b t l -> (b t) l")
        text_embed = self.first_stage_model.model.encode_text_embed(text_tokens)
        text_embed = rearrange(text_embed, "(b t) d -> b t d", b=b)
        text_embed = text_embed.mean(dim=1)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        return text_embed

    @torch.no_grad()
    def encode_image(self, img):

        return self.first_stage_model.model.encode_image_embed(img)

    @torch.no_grad()
    def encode_surface(self, surface):

        return self.first_stage_model.model.encode_shape_embed(surface, return_latents=False)

    @torch.no_grad()
    def empty_text_cond(self, cond):

        return torch.zeros_like(cond, device=cond.device)

    @torch.no_grad()
    def empty_img_cond(self, cond):

        return torch.zeros_like(cond, device=cond.device)

    @torch.no_grad()
    def empty_surface_cond(self, cond):

        return torch.zeros_like(cond, device=cond.device)

    @torch.no_grad()
    def encode_first_stage(self, surface: torch.FloatTensor, sample_posterior=True):

        z_q = self.first_stage_model.encode(surface, sample_posterior)
        z_q = self.z_scale_factor * z_q

        return z_q

    @torch.no_grad()
    def decode_first_stage(self, z_q: torch.FloatTensor, **kwargs):

        z_q = 1. / self.z_scale_factor * z_q
        latents = self.first_stage_model.decode(z_q, **kwargs)
        return latents

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
                and batch_idx == 0 and self.ckpt_path is None:
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")

            z_q = self.encode_first_stage(batch[self.first_stage_key])
            z = z_q.detach()

            del self.z_scale_factor
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")

            print("### USING STD-RESCALING ###")

    def compute_loss(self, model_outputs, split):
        """

        Args:
            model_outputs (dict):
                - x_0:
                - noise:
                - noise_prior:
                - noise_pred:
                - noise_pred_prior:

            split (str):

        Returns:

        """

        pred = model_outputs["pred"]

        if self.noise_scheduler.prediction_type == "epsilon":
            target = model_outputs["noise"]
        elif self.noise_scheduler.prediction_type == "sample":
            target = model_outputs["x_0"]
        else:
            raise NotImplementedError(f"Prediction Type: {self.noise_scheduler.prediction_type} not yet supported.")

        if self.loss_cfg.loss_type == "l1":
            simple = F.l1_loss(pred, target, reduction="mean")
        elif self.loss_cfg.loss_type in ["mse", "l2"]:
            simple = F.mse_loss(pred, target, reduction="mean")
        else:
            raise NotImplementedError(f"Loss Type: {self.loss_cfg.loss_type} not yet supported.")

        total_loss = simple

        loss_dict = {
            f"{split}/total_loss": total_loss.clone().detach(),
            f"{split}/simple": simple.detach(),
        }

        return total_loss, loss_dict

    def forward(self, batch):
        """

        Args:
            batch:

        Returns:

        """

        if self.first_stage_model is None:
            self.instantiate_first_stage(self.first_stage_config)

        latents = self.encode_first_stage(batch[self.first_stage_key])

        # conditions = self.cond_stage_model.encode(batch[self.cond_stage_key])

        conditions = self.cond_stage_model[self.cond_stage_key](batch[self.cond_stage_key]).unsqueeze(1)

        mask = torch.rand((len(conditions), 1, 1), device=conditions.device, dtype=conditions.dtype) >= 0.1
        conditions = conditions * mask.to(conditions)

        # Sample noise that we"ll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # diffusion model forward
        noise_pred = self.model(noisy_z, timesteps, conditions)

        diffusion_outputs = {
            "x_0": noisy_z,
            "noise": noise,
            "pred": noise_pred
        }

        return diffusion_outputs

    def training_step(self, batch: Dict[str, Union[torch.FloatTensor, List[str]]],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """

        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor):
                - image (torch.FloatTensor): if provide, [bs, 3, h, w], item range [0, 1]
                - depth (torch.FloatTensor): if provide, [bs, 1, h, w], item range [-1, 1]
                - normal (torch.FloatTensor): if provide, [bs, 3, h, w], item range [-1, 1]
                - text (list of str):

            batch_idx (int):

            optimizer_idx (int):

        Returns:
            loss (torch.FloatTensor):

        """

        diffusion_outputs = self(batch)

        loss, loss_dict = self.compute_loss(diffusion_outputs, "train")
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.FloatTensor],
                        batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """

        Args:
            batch (dict): the batch sample, and it contains:
                - surface_pc (torch.FloatTensor): [n_pts, 4]
                - surface_feats (torch.FloatTensor): [n_pts, c]
                - text (list of str):

            batch_idx (int):

            optimizer_idx (int):

        Returns:
            loss (torch.FloatTensor):

        """

        diffusion_outputs = self(batch)

        loss, loss_dict = self.compute_loss(diffusion_outputs, "val")
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    @torch.no_grad()
    def sample(self,
               batch: Dict[str, Union[torch.FloatTensor, List[str]]],
               sample_times: int = 1,
               steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               eta: float = 0.0,
               return_intermediates: bool = False, **kwargs):

        if self.first_stage_model is None:
            self.instantiate_first_stage(self.first_stage_config)

        if steps is None:
            steps = self.scheduler_cfg.num_inference_steps

        if guidance_scale is None:
            guidance_scale = self.scheduler_cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale > 0

        # conditional encode
        xc = batch[self.cond_stage_key]
        # cond = self.cond_stage_model[self.cond_stage_key](xc)
        cond = self.cond_stage_model[self.cond_stage_key](xc).unsqueeze(1)

        if do_classifier_free_guidance:
            """
            Note: There are two kinds of uncond for text. 
            1: using "" as uncond text; (in SAL diffusion)
            2: zeros_like(cond) as uncond text; (in MDM)
            """
            # un_cond = self.cond_stage_model.unconditional_embedding(batch_size=len(xc))
            un_cond = self.cond_stage_model[f"{self.cond_stage_key}_unconditional_embedding"](cond)
            # un_cond = torch.zeros_like(cond, device=cond.device)
            cond = torch.cat([un_cond, cond], dim=0)

        outputs = []
        latents = None

        if not return_intermediates:
            for _ in range(sample_times):
                sample_loop = ddim_sample(
                    self.denoise_scheduler,
                    self.model,
                    shape=self.first_stage_model.latent_shape,
                    cond=cond,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    device=self.device,
                    eta=eta,
                    disable_prog=not self.zero_rank
                )
                for sample, t in sample_loop:
                    latents = sample
                outputs.append(self.decode_first_stage(latents, **kwargs))
        else:

            sample_loop = ddim_sample(
                self.denoise_scheduler,
                self.model,
                shape=self.first_stage_model.latent_shape,
                cond=cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=not self.zero_rank
            )

            iter_size = steps // sample_times
            i = 0
            for sample, t in sample_loop:
                latents = sample
                if i % iter_size == 0 or i == steps - 1:
                    outputs.append(self.decode_first_stage(latents, **kwargs))
                i += 1

        return outputs
