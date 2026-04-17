#!/usr/bin/env python3
"""
Generate videos using PRE-QUANTIZED bitsandbytes NF4 models.

This version is modified to avoid host-RAM OOM by:
- NOT loading both diffusion models in __init__
- Loading diffusion models lazily the first time they are needed
"""

import argparse
import gc
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent))

from load_prequant import load_quantized_model
from wan.configs.wan_i2v_A14B import i2v_A14B as cfg
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WanI2V_PreQuant:
    """Image-to-video pipeline using pre-quantized NF4 models."""

    def __init__(
        self,
        checkpoint_dir: str,
        device_id: int = 0,
        t5_cpu: bool = True,
        t5_device_str: str = "cpu",
        # vae_device_str: str = None,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        # self.vae_device = torch.device(vae_device_str) if vae_device_str else self.device
        
        self.config = cfg
        self.t5_cpu = t5_cpu
        self.t5_device = torch.device(t5_device_str)
        self._cached_prompt = None
        self._cached_context = None
        self._cached_context_null = None
        self.num_train_timesteps = cfg.num_train_timesteps
        self.boundary = cfg.boundary
        self.param_dtype = cfg.param_dtype
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size
        self.sample_neg_prompt = cfg.sample_neg_prompt

        logger.info("Loading T5 encoder on %s...", t5_device_str)
        local_tokenizer = os.path.join(checkpoint_dir, "tokenizer")
        tokenizer_path = local_tokenizer if os.path.isdir(local_tokenizer) else cfg.t5_tokenizer
        self.text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=self.t5_device,
            checkpoint_path=os.path.join(checkpoint_dir, cfg.t5_checkpoint),
            tokenizer_path=tokenizer_path,
            shard_fn=None,
        )

        logger.info("Loading VAE...")
        # logger.info("Loading VAE on %s...", self.vae_device)
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, cfg.vae_checkpoint),
            device=self.device,
        )

        logger.info("Preparing pre-quantized NF4 diffusion model paths...")

        self.low_noise_dir = os.path.join(checkpoint_dir, cfg.low_noise_checkpoint + "_bnb_nf4")
        self.high_noise_dir = os.path.join(checkpoint_dir, cfg.high_noise_checkpoint + "_bnb_nf4")

        for d in [self.low_noise_dir, self.high_noise_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Pre-quantized model not found: {d}\n"
                    "Expected packaged NF4 directories."
                )

        self.low_noise_model: Optional[torch.nn.Module] = None
        self.high_noise_model: Optional[torch.nn.Module] = None

        logger.info("Init complete (diffusion models will lazy-load).")


    def _lazy_load_low(self) -> torch.nn.Module:
        if self.low_noise_model is None:
            logger.info("Lazy-loading low_noise_model (CPU)...")
            self.low_noise_model = load_quantized_model(self.low_noise_dir, device="cpu")
            logger.info("low_noise_model loaded.")
        return self.low_noise_model

    def _lazy_load_high(self) -> torch.nn.Module:
        if self.high_noise_model is None:
            logger.info("Lazy-loading high_noise_model (CPU)...")
            self.high_noise_model = load_quantized_model(self.high_noise_dir, device="cpu")
            logger.info("high_noise_model loaded.")
        return self.high_noise_model

    def _prepare_model_for_timestep(self, t, boundary):
        if t.item() >= boundary:
            required = self._lazy_load_high()
        else:
            required = self._lazy_load_low()
        
        try:
            if next(required.parameters()).device.type == "cpu":
                required.to(self.device)
        except StopIteration:
            pass
        
        return required

    def generate(
        self,
        input_prompt: str,
        img: Image.Image,
        raw_init_tensor: Optional[torch.Tensor] = None,
        action_path: str = None,
        max_area: int = 720 * 1280,
        frame_num: int = 81,
        shift: float = 5.0,
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
    ):
        """
        Generate video from image and text prompt.

        Args:
            raw_init_tensor: Optional raw tensor from previous chunk's VAE output.
                             Shape [C, 1, H, W] in [-1, 1] range. If provided, used
                             instead of img to avoid JPEG quality loss between chunks.
        """
        c2ws = None
        use_camera_control = False

        if action_path is not None:
            poses_path = os.path.join(action_path, "poses.npy")
            intrinsics_path = os.path.join(action_path, "intrinsics.npy")

            if not os.path.exists(poses_path) or not os.path.exists(intrinsics_path):
                logger.warning(
                    "Skipping camera conditioning: missing poses.npy or intrinsics.npy: %s",
                    action_path,
                )
            else:
                c2ws = np.load(poses_path)
                len_c2ws = len(c2ws)

                if len_c2ws == 0:
                    logger.warning("Skipping camera conditioning: poses.npy is empty")
                else:
                    frame_num = min(frame_num, len_c2ws)
                    c2ws = c2ws[:frame_num]

                    if len(c2ws) >= 2 and frame_num >= 2:
                        use_camera_control = True
                    else:
                        logger.warning(
                            "Skipping camera conditioning: only %d pose(s) for frame_num=%d",
                            len(c2ws), frame_num,
                        )

        guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale

        # Use raw tensor if available (avoids JPEG quality loss between chunks)
        if raw_init_tensor is not None:
            # raw_init_tensor is [C, 1, H, W] in [-1, 1] range from VAE output
            img_tensor = raw_init_tensor.squeeze(1).to(self.device)  # [C, H, W]
        else:
            img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img_tensor.shape[1:]
        aspect_ratio = h / w

        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
            dim=1,
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Text encoding with cache
        if self._cached_prompt != (input_prompt, n_prompt):
            context = self.text_encoder([input_prompt], self.t5_device)
            context_null = self.text_encoder([n_prompt], self.t5_device)
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            self._cached_prompt = (input_prompt, n_prompt)
            self._cached_context = context
            self._cached_context_null = context_null
        else:
            context = self._cached_context
            context_null = self._cached_context_null

        # Camera preparation
        logger.info("use_camera_control=%s, frame_num=%d", use_camera_control, frame_num)

        dit_cond_dict = None
        if use_camera_control:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks, 480, 832, h, w, h, w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)

            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(
                    0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1
                ),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                "b (f h w) c -> b c f h w",
                f=lat_f,
                h=lat_h,
                w=lat_w,
            ).to(self.param_dtype)
            dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

        # Encode image
        y = self.vae.encode(
            [
                torch.concat([
                    torch.nn.functional.interpolate(
                        img_tensor[None].cpu(), size=(h, w), mode="bicubic",
                    ).transpose(0, 1),
                    torch.zeros(3, F - 1, h, w),
                ], dim=1).to(self.device)
            ]
        )[0]
        y = torch.concat([msk, y])

        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            boundary = self.boundary * self.num_train_timesteps

            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                sampling_steps,
                device=self.device,
                shift=shift,
            )
            timesteps = sample_scheduler.timesteps

            # Determine which steps get CFG (first half only for low guide scales)
            # This saves ~30% compute vs full CFG while maintaining quality
            num_cfg_steps = len(timesteps)
            if guide_scale[0] <= 3.0 and guide_scale[0] > 1.0:
                num_cfg_steps = max(1, len(timesteps) // 2)

            latent = noise

            arg_c = {
                "context": [context[0]],
                "seq_len": max_seq_len,
                "y": [y],
                "dit_cond_dict": dit_cond_dict,
            }
            arg_null = {
                "context": context_null,
                "seq_len": max_seq_len,
                "y": [y],
                "dit_cond_dict": dit_cond_dict,
            }

            torch.cuda.empty_cache()

            self._prepare_model_for_timestep(timesteps[0], boundary)
            logger.info("Prepared first model on GPU for sampling")

            for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling")):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                model = self._prepare_model_for_timestep(t, boundary)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                # First-half CFG: only run unconditional pass on early steps
                use_cfg_this_step = (
                    sample_guide_scale > 1.0 and step_idx < num_cfg_steps
                )

                if not use_cfg_this_step or sample_guide_scale <= 1.0:
                    noise_pred = model(latent_model_input, t=timestep, **arg_c)[0]
                else:
                    noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                    noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latent = temp_x0.squeeze(0)

            torch.cuda.empty_cache()

            # Move latent to VAE device for decode, then back
            # latent_for_decode = latent.to(self.vae_device)
            # videos = self.vae.decode([latent.to(self.vae_device)])
            videos = self.vae.decode([latent])


        del noise, latent
        gc.collect()
        torch.cuda.synchronize()

        return videos[0]


def save_video(frames: torch.Tensor, output_path: str, fps: int = 16):
    import imageio
    frames = ((frames + 1) / 2 * 255).clamp(0, 255).byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264")
    logger.info("Saved video to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate videos with pre-quantized NF4 models")
    script_dir = str(Path(__file__).parent)
    parser.add_argument("--ckpt_dir", type=str, default=script_dir)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--action_path", type=str, default=None)
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--t5_cpu", action="store_true", default=True)  # keep behavior
    args = parser.parse_args()

    h, w = map(int, args.size.split("*"))
    max_area = h * w

    img = Image.open(args.image).convert("RGB")

    pipeline = WanI2V_PreQuant(checkpoint_dir=args.ckpt_dir, t5_cpu=args.t5_cpu)

    logger.info("Generating video...")
    video = pipeline.generate(
        input_prompt=args.prompt,
        img=img,
        action_path=args.action_path,
        max_area=max_area,
        frame_num=args.frame_num,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )

    save_video(video, args.output)


if __name__ == "__main__":
    main()