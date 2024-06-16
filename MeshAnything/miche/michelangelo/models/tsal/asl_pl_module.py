# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler
from typing import Union
from functools import partial

from MeshAnything.miche.michelangelo.utils import instantiate_from_config

from .tsal_base import (
    AlignedShapeAsLatentModule,
    ShapeAsLatentModule,
    Latent2MeshOutput,
    AlignedMeshOutput
)
from MeshAnything.miche.michelangelo.models.tsal.inference_utils import extract_geometry
import trimesh

class AlignedShapeAsLatentPLModule(nn.Module):
    def __init__(self, *,
                 shape_module_cfg,
                 aligned_module_cfg,
                 loss_cfg,
                 optimizer_cfg: Optional[DictConfig] = None,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        shape_model: ShapeAsLatentModule = instantiate_from_config(
            shape_module_cfg, device=None, dtype=None
        )
        self.model: AlignedShapeAsLatentModule = instantiate_from_config(
            aligned_module_cfg, shape_model=shape_model
        )

        self.loss = instantiate_from_config(loss_cfg)

        self.optimizer_cfg = optimizer_cfg

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def set_shape_model_only(self):
        self.model.set_shape_model_only()



    @property
    def latent_shape(self):
        return self.model.shape_model.latent_shape

    @property
    def zero_rank(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return zero_rank

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

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        trainable_parameters = list(self.model.parameters())

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

    def forward(self,
                surface: torch.FloatTensor,
                image: torch.FloatTensor,
                text: torch.FloatTensor,
                volume_queries: torch.FloatTensor):

        """

        Args:
            surface (torch.FloatTensor):
            image (torch.FloatTensor):
            text (torch.FloatTensor):
            volume_queries (torch.FloatTensor):

        Returns:

        """

        embed_outputs, shape_z = self.model(surface, image, text)

        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)
        latents = self.model.shape_model.decode(shape_zq)
        logits = self.model.shape_model.query_geometry(volume_queries, latents)

        return embed_outputs, logits, posterior

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):

        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        shape_embed, shape_zq, posterior = self.model.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return shape_zq

    def encode_latents(self, surface: torch.FloatTensor):

        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        shape_embed, shape_latents = self.model.shape_model.encode_latents(
            pc=pc, feats=feats
        )
        shape_embed = shape_embed.unsqueeze(1)
        assert shape_embed.shape[1] == 1 and shape_latents.shape[1] == 256
        cat_latents = torch.cat([shape_embed, shape_latents], dim=1)

        return cat_latents

    def recon(self, surface):
        cat_latents = self.encode_latents(surface)
        shape_latents = cat_latents[:, 1:]
        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_latents)

        # decoding
        latents = self.model.shape_model.decode(shape_zq)
        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

        # reconstruction
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=surface.device,
            batch_size=surface.shape[0],
            bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
            octree_depth=7,
            num_chunks=10000,
        )
        recon_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])

        return recon_mesh


    def to_shape_latents(self, latents):

        shape_zq, posterior = self.model.shape_model.encode_kl_embed(latents, sample_posterior = False)
        return self.model.shape_model.decode(shape_zq)

    def decode(self,
               z_q,
               bounds: Union[Tuple[float], List[float], float] = 1.1,
               octree_depth: int = 7,
               num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        latents = self.model.shape_model.decode(z_q)  # latents: [bs, num_latents, dim]
        outputs = self.latent2mesh(latents, bounds=bounds, octree_depth=octree_depth, num_chunks=num_chunks)

        return outputs

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """

        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor): [bs, n_surface, (3 + input_dim)]
                - image (torch.FloatTensor): [bs, 3, 224, 224]
                - text (torch.FloatTensor): [bs, num_templates, 77]
                - geo_points (torch.FloatTensor): [bs, n_pts, (3 + 1)]

            batch_idx (int):

            optimizer_idx (int):

        Returns:
            loss (torch.FloatTensor):

        """

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels = batch["geo_points"][..., -1]

        embed_outputs, shape_logits, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            shape_logits=shape_logits,
            shape_labels=shape_labels,
            split="train"
        )

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=shape_logits.shape[0],
                      sync_dist=False, rank_zero_only=True)

        return aeloss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels = batch["geo_points"][..., -1]

        embed_outputs, shape_logits, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            shape_logits=shape_logits,
            shape_labels=shape_labels,
            split="val"
        )
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=shape_logits.shape[0],
                      sync_dist=False, rank_zero_only=True)

        return aeloss

    def visual_alignment(self,
                         surface: torch.FloatTensor,
                         image: torch.FloatTensor,
                         text: torch.FloatTensor,
                         description: Optional[List[str]] = None,
                         bounds: Union[Tuple[float], List[float]] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                         octree_depth: int = 7,
                         num_chunks: int = 10000) -> List[AlignedMeshOutput]:

        """

        Args:
            surface:
            image:
            text:
            description:
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[AlignedMeshOutput]): the mesh outputs list.

        """

        outputs = []

        device = surface.device
        bs = surface.shape[0]

        embed_outputs, shape_z = self.model(surface, image, text)

        # calculate the similarity
        image_embed = embed_outputs["image_embed"]
        text_embed = embed_outputs["text_embed"]
        shape_embed = embed_outputs["shape_embed"]

        # normalized features
        shape_embed = F.normalize(shape_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # B x B
        shape_text_similarity = (100.0 * shape_embed @ text_embed.T).softmax(dim=-1)

        # B x B
        shape_image_similarity = (100.0 * shape_embed @ image_embed.T).softmax(dim=-1)

        # shape reconstruction
        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)
        latents = self.model.shape_model.decode(shape_zq)
        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

        # 2. decode geometry
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=bs,
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = AlignedMeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            out.surface = surface[i].cpu().numpy()
            out.image = image[i].cpu().numpy()
            if description is not None:
                out.text = description[i]
            out.shape_text_similarity = shape_text_similarity[i, i]
            out.shape_image_similarity = shape_image_similarity[i, i]

            outputs.append(out)

        return outputs

    def latent2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        """

        Args:
            latents: [bs, num_latents, dim]
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[MeshOutput]): the mesh outputs list.

        """

        outputs = []

        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f

            outputs.append(out)

        return outputs

