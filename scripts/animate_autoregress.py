import argparse
import datetime
import random
import os
import logging
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from consisti2v.models.videoldm_unet import VideoLDMUNet3DConditionModel
from consisti2v.pipelines.pipeline_autoregress_animation import AutoregressiveAnimationPipeline
from consisti2v.utils.util import save_videos_grid
from diffusers.utils.import_utils import is_xformers_available

def main(args, config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    diffusers.utils.logging.set_verbosity_info()

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"{config.output_dir}/{config.output_name}-{time_str}"
    os.makedirs(savedir)

    samples = []
    sample_idx = 0

    ### >>> create validation pipeline >>> ###
    if config.pipeline_pretrained_path is None:
        noise_scheduler = DDIMScheduler(**OmegaConf.to_container(config.noise_scheduler_kwargs))
        tokenizer       = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer", use_safetensors=True)
        text_encoder    = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
        vae             = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae", use_safetensors=True)            
        unet            = VideoLDMUNet3DConditionModel.from_pretrained(
            config.pretrained_model_path,
            subfolder="unet",
            variant=config.unet_additional_kwargs['variant'],
            temp_pos_embedding=config.unet_additional_kwargs['temp_pos_embedding'],
            augment_temporal_attention=config.unet_additional_kwargs['augment_temporal_attention'],
            use_temporal=True,
            n_frames=config.sampling_kwargs['n_frames'],
            n_temp_heads=config.unet_additional_kwargs['n_temp_heads'],
            first_frame_condition_mode=config.unet_additional_kwargs['first_frame_condition_mode'],
            use_frame_stride_condition=config.unet_additional_kwargs['use_frame_stride_condition'],
            use_safetensors=True
        )

        params_unet = [p.numel() for n, p in unet.named_parameters()]
        params_vae = [p.numel() for n, p in vae.named_parameters()]
        params_text_encoder = [p.numel() for n, p in text_encoder.named_parameters()]
        params = params_unet + params_vae + params_text_encoder
        print(f"### UNet Parameters: {sum(params) / 1e6} M")

        # 1. unet ckpt
        if config.unet_path is not None:
            if os.path.isdir(config.unet_path):
                unet_dict = VideoLDMUNet3DConditionModel.from_pretrained(config.unet_path)
                m, u = unet.load_state_dict(unet_dict.state_dict(), strict=False)
                assert len(u) == 0
                del unet_dict
            else:
                checkpoint_dict = torch.load(config.unet_path, map_location="cpu")
                state_dict = checkpoint_dict["state_dict"] if "state_dict" in checkpoint_dict else checkpoint_dict
                if config.unet_ckpt_prefix is not None:
                    state_dict = {k.replace(config.unet_ckpt_prefix, ''): v for k, v in state_dict.items()}
                m, u = unet.load_state_dict(state_dict, strict=False)
                assert len(u) == 0

        if is_xformers_available() and int(torch.__version__.split(".")[0]) < 2:
            unet.enable_xformers_memory_efficient_attention()

        pipeline = AutoregressiveAnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=noise_scheduler)
    
    else:
        pipeline = AutoregressiveAnimationPipeline.from_pretrained(config.pipeline_pretrained_path)

    pipeline.to("cuda")

    # (frameinit) initialize frequency filter for noise reinitialization -------------
    if config.frameinit_kwargs.enable:
        pipeline.init_filter(
            width         = config.sampling_kwargs.width,
            height        = config.sampling_kwargs.height,
            video_length  = config.sampling_kwargs.n_frames,
            filter_params = config.frameinit_kwargs.filter_params,
        )
    # -------------------------------------------------------------------------------
    ### <<< create validation pipeline <<< ###

    if args.prompt is not None:
        prompts = [args.prompt]
        n_prompts = [args.n_prompt]
        first_frame_paths = [args.path_to_first_frame]
        random_seeds = [int(args.seed)] if args.seed != "random" else "random"
    else:
        prompt_config = OmegaConf.load(args.prompt_config)
        prompts = prompt_config.prompts
        n_prompts = list(prompt_config.n_prompts) * len(prompts) if len(prompt_config.n_prompts) == 1 else prompt_config.n_prompts
        first_frame_paths = prompt_config.path_to_first_frames
        random_seeds = prompt_config.seeds
    
    if random_seeds == "random":
        random_seeds = [random.randint(0, 1e5) for _ in range(len(prompts))]
    else:
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
    
    config.prompt_kwargs = OmegaConf.create({"random_seeds": [], "prompts": prompts, "n_prompts": n_prompts, "first_frame_paths": first_frame_paths})
    for prompt_idx, (prompt, n_prompt, first_frame_path, random_seed) in enumerate(zip(prompts, n_prompts, first_frame_paths, random_seeds)):
        # manually set random seed for reproduction
        if random_seed != -1: torch.manual_seed(random_seed)
        else: torch.seed()
        config.prompt_kwargs.random_seeds.append(torch.initial_seed())
        
        print(f"current seed: {torch.initial_seed()}")
        print(f"sampling {prompt} ...")
        sample = pipeline(
            prompt,
            negative_prompt       = n_prompt,
            first_frame_paths     = first_frame_path,
            num_inference_steps   = config.sampling_kwargs.steps,
            guidance_scale_txt    = config.sampling_kwargs.guidance_scale_txt,
            guidance_scale_img    = config.sampling_kwargs.guidance_scale_img,
            width                 = config.sampling_kwargs.width,
            height                = config.sampling_kwargs.height,
            video_length          = config.sampling_kwargs.n_frames,
            noise_sampling_method = config.unet_additional_kwargs['noise_sampling_method'],
            noise_alpha           = float(config.unet_additional_kwargs['noise_alpha']),
            eta                   = config.sampling_kwargs.ddim_eta,
            frame_stride          = config.sampling_kwargs.frame_stride,
            guidance_rescale      = config.sampling_kwargs.guidance_rescale,
            num_videos_per_prompt = config.sampling_kwargs.num_videos_per_prompt,
            autoregress_steps     = config.sampling_kwargs.autoregress_steps,
            use_frameinit          = config.frameinit_kwargs.enable,
            frameinit_noise_level  = config.frameinit_kwargs.noise_level,
        ).videos
        samples.append(sample)

        prompt = "-".join((prompt.replace("/", "").split(" ")[:10])).replace(":", "")
        if sample.shape[0] > 1:
            for cnt, samp in enumerate(sample):
                save_videos_grid(samp.unsqueeze(0), f"{savedir}/sample/{sample_idx}-{cnt + 1}-{prompt}.{args.format}", format=args.format)
        else:
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.{args.format}", format=args.format)
        print(f"save to {savedir}/sample/{prompt}.{args.format}")
        
        sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.{args.format}", n_rows=4, format=args.format)

    OmegaConf.save(config, f"{savedir}/config.yaml")

    if args.save_model:
        pipeline.save_pretrained(f"{savedir}/model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/inference_autoregress.yaml")
    parser.add_argument("--prompt", "-p", type=str, default=None)
    parser.add_argument("--n_prompt", "-n", type=str, default="")
    parser.add_argument("--seed", type=str, default="random")
    parser.add_argument("--path_to_first_frame", "-f", type=str, default=None)
    parser.add_argument("--prompt_config", type=str, default="configs/prompts/default.yaml")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4"])
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("optional_args", nargs='*', default=[])
    args = parser.parse_args()

    config = OmegaConf.load(args.inference_config)

    if args.optional_args:
        modified_config = OmegaConf.from_dotlist(args.optional_args)
        config = OmegaConf.merge(config, modified_config)

    main(args, config)
