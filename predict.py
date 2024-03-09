# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import subprocess
from omegaconf import OmegaConf
import torch
from cog import BasePredictor, Input, Path
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from consisti2v.models.videoldm_unet import VideoLDMUNet3DConditionModel
from consisti2v.pipelines.pipeline_conditional_animation import (
    ConditionalAnimationPipeline,
)
from consisti2v.utils.util import save_videos_grid


URL = {
    k: f"https://weights.replicate.delivery/default/ConsistI2V_cache/{k}.tar"
    for k in ["text_encoder", "vae", "tokenizer", "unet"]
}
MODEL_CACHE = {
    k: f"model_cache/{k}" for k in ["text_encoder", "vae", "tokenizer", "unet"]
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        inference_config = "configs/inference/inference.yaml"
        self.config = OmegaConf.load(inference_config)
        noise_scheduler = DDIMScheduler(
            **OmegaConf.to_container(self.config.noise_scheduler_kwargs)
        )

        # The weights are pushed to replicate.delivery, see def save_weights() below for details
        for k in ["text_encoder", "vae", "tokenizer", "unet"]:
            if not os.path.exists(MODEL_CACHE[k]):
                download_weights(URL[k], MODEL_CACHE[k])

        tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_CACHE["tokenizer"], use_safetensors=True
        )
        text_encoder = CLIPTextModel.from_pretrained(MODEL_CACHE["text_encoder"])
        vae = AutoencoderKL.from_pretrained(MODEL_CACHE["vae"], use_safetensors=True)
        unet = VideoLDMUNet3DConditionModel.from_pretrained(
            MODEL_CACHE["unet"],
            subfolder="unet",
            variant=self.config.unet_additional_kwargs["variant"],
            temp_pos_embedding=self.config.unet_additional_kwargs["temp_pos_embedding"],
            augment_temporal_attention=self.config.unet_additional_kwargs[
                "augment_temporal_attention"
            ],
            use_temporal=True,
            n_frames=self.config.sampling_kwargs["n_frames"],
            n_temp_heads=self.config.unet_additional_kwargs["n_temp_heads"],
            first_frame_condition_mode=self.config.unet_additional_kwargs[
                "first_frame_condition_mode"
            ],
            use_frame_stride_condition=self.config.unet_additional_kwargs[
                "use_frame_stride_condition"
            ],
            use_safetensors=True,
        )

        self.pipeline = ConditionalAnimationPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
        ).to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image as the first frame of the video."),
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        text_guidance_scale: float = Input(
            description="Scale for classifier-free guidance from the text",
            ge=1,
            le=50,
            default=7.5,
        ),
        image_guidance_scale: float = Input(
            description="Scale for classifier-free guidance from the image", default=1.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        if self.config.frameinit_kwargs.enable:
            self.pipeline.init_filter(
                width=self.config.sampling_kwargs.width,
                height=self.config.sampling_kwargs.height,
                video_length=self.config.sampling_kwargs.n_frames,
                filter_params=self.config.frameinit_kwargs.filter_params,
            )

        sample = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            first_frame_paths=str(image),
            num_inference_steps=num_inference_steps,
            guidance_scale_txt=text_guidance_scale,
            guidance_scale_img=image_guidance_scale,
            width=self.config.sampling_kwargs.width,  # output video only supports 16 frames of 256x256
            height=self.config.sampling_kwargs.height,
            video_length=self.config.sampling_kwargs.n_frames,
            noise_sampling_method=self.config.unet_additional_kwargs[
                "noise_sampling_method"
            ],
            noise_alpha=float(self.config.unet_additional_kwargs["noise_alpha"]),
            eta=self.config.sampling_kwargs.ddim_eta,
            frame_stride=self.config.sampling_kwargs.frame_stride,
            guidance_rescale=self.config.sampling_kwargs.guidance_rescale,
            num_videos_per_prompt=self.config.sampling_kwargs.num_videos_per_prompt,
            use_frameinit=self.config.frameinit_kwargs.enable,
            frameinit_noise_level=self.config.frameinit_kwargs.noise_level,
            camera_motion=self.config.frameinit_kwargs.camera_motion,
        ).videos
        out_path = "/tmp/out.mp4"
        save_videos_grid(sample, out_path, format="mp4")
        return Path(out_path)


def save_weights():
    "Load the weights, saved to local and push to replicate.delivery"
    inference_config = "configs/inference/inference.yaml"
    config = OmegaConf.load(inference_config)

    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_path, subfolder="tokenizer", use_safetensors=True
    )
    tokenizer.save_pretrained("ConsistI2V_cache/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        config.pretrained_model_path, subfolder="text_encoder"
    )
    text_encoder.save_pretrained(
        "ConsistI2V_cache/text_encoder", safe_serialization=True
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_path, subfolder="vae", use_safetensors=True
    )
    vae.save_pretrained("ConsistI2V_cache/vae", safe_serialization=True)
    unet = VideoLDMUNet3DConditionModel.from_pretrained(
        config.pretrained_model_path,
        subfolder="unet",
        variant=config.unet_additional_kwargs["variant"],
        temp_pos_embedding=config.unet_additional_kwargs["temp_pos_embedding"],
        augment_temporal_attention=config.unet_additional_kwargs[
            "augment_temporal_attention"
        ],
        use_temporal=True,
        n_frames=config.sampling_kwargs["n_frames"],
        n_temp_heads=config.unet_additional_kwargs["n_temp_heads"],
        first_frame_condition_mode=config.unet_additional_kwargs[
            "first_frame_condition_mode"
        ],
        use_frame_stride_condition=config.unet_additional_kwargs[
            "use_frame_stride_condition"
        ],
        use_safetensors=True,
    )
    unet.save_pretrained("ConsistI2V_cache/unet", safe_serialization=True)
