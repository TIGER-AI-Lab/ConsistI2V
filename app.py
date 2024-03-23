
import os
import json
import torch
import random
import requests
from PIL import Image
import numpy as np

import gradio as gr
from datetime import datetime

import torchvision.transforms as T

from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from consisti2v.pipelines.pipeline_conditional_animation import ConditionalAnimationPipeline
from consisti2v.utils.util import save_videos_grid
from omegaconf import OmegaConf


sample_idx     = 0
scheduler_dict = {
    "DDIM": DDIMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

class AnimateController:
    def __init__(self):
        
        # config dirs
        self.basedir        = os.getcwd()
        self.savedir        = os.path.join(self.basedir, "samples/Gradio", datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.image_resolution = (256, 256)
        # config models
        self.pipeline = ConditionalAnimationPipeline.from_pretrained("TIGER-Lab/ConsistI2V", torch_dtype=torch.float16,)
        self.pipeline.to("cuda")

    def update_textbox_and_save_image(self, input_image, height_slider, width_slider, center_crop):
        pil_image = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")
        img_path = os.path.join(self.savedir, "input_image.png")
        pil_image.save(img_path)
        self.image_resolution = pil_image.size
        original_width, original_height = pil_image.size
        if center_crop:
            crop_aspect_ratio = width_slider / height_slider
            aspect_ratio = original_width / original_height
            if aspect_ratio > crop_aspect_ratio:
                new_width = int(crop_aspect_ratio * original_height)
                left = (original_width - new_width) / 2
                top = 0
                right = left + new_width
                bottom = original_height
                pil_image = pil_image.crop((left, top, right, bottom))
            elif aspect_ratio < crop_aspect_ratio:
                new_height = int(original_width / crop_aspect_ratio)
                top = (original_height - new_height) / 2
                left = 0
                right = original_width
                bottom = top + new_height
                pil_image = pil_image.crop((left, top, right, bottom))
                
        pil_image = pil_image.resize((width_slider, height_slider))
        return gr.Textbox.update(value=img_path), gr.Image.update(value=np.array(pil_image))

    def animate(
        self,
        prompt_textbox, 
        negative_prompt_textbox,
        input_image_path,
        sampler_dropdown, 
        sample_step_slider, 
        width_slider, 
        height_slider, 
        txt_cfg_scale_slider,
        img_cfg_scale_slider,
        center_crop,
        frame_stride,
        use_frameinit,
        frame_init_noise_level,
        seed_textbox
    ):
        if self.pipeline is None:
            raise gr.Error(f"Please select a pretrained pipeline path.")
        if input_image_path == "":
            raise gr.Error(f"Please upload an input image.")
        if (not center_crop) and (width_slider % 8 != 0 or height_slider % 8 != 0):
            raise gr.Error(f"`height` and `width` have to be divisible by 8 but are {height_slider} and {width_slider}.")
        if center_crop and (width_slider % 8 != 0 or height_slider % 8 != 0):
            raise gr.Error(f"`height` and `width` (after cropping) have to be divisible by 8 but are {height_slider} and {width_slider}.")

        if is_xformers_available() and int(torch.__version__.split(".")[0]) < 2: self.pipeline.unet.enable_xformers_memory_efficient_attention()

        if seed_textbox != -1 and seed_textbox != "": torch.manual_seed(int(seed_textbox))
        else: torch.seed()
        seed = torch.initial_seed()

        if input_image_path.startswith("http://") or input_image_path.startswith("https://"):
            first_frame = Image.open(requests.get(input_image_path, stream=True).raw).convert('RGB')
        else:
            first_frame = Image.open(input_image_path).convert('RGB')
        
        original_width, original_height = first_frame.size

        if not center_crop:
            img_transform = T.Compose([
                T.ToTensor(),
                T.Resize((height_slider, width_slider), antialias=None),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        else:
            aspect_ratio = original_width / original_height
            crop_aspect_ratio = width_slider / height_slider
            if aspect_ratio > crop_aspect_ratio:
                center_crop_width = int(crop_aspect_ratio * original_height)
                center_crop_height = original_height
            elif aspect_ratio < crop_aspect_ratio:
                center_crop_width = original_width
                center_crop_height = int(original_width / crop_aspect_ratio)
            else:
                center_crop_width = original_width
                center_crop_height = original_height
            img_transform = T.Compose([
                T.ToTensor(),
                T.CenterCrop((center_crop_height, center_crop_width)),
                T.Resize((height_slider, width_slider), antialias=None),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        
        first_frame = img_transform(first_frame).unsqueeze(0)
        first_frame = first_frame.to("cuda")

        if use_frameinit:
            self.pipeline.init_filter(
                width         = width_slider,
                height        = height_slider,
                video_length  = 16,
                filter_params = OmegaConf.create({'method': 'gaussian', 'd_s': 0.25, 'd_t': 0.25,})
            )


        sample = self.pipeline(
            prompt_textbox,
            negative_prompt       = negative_prompt_textbox,
            first_frames          = first_frame,
            num_inference_steps   = sample_step_slider,
            guidance_scale_txt    = txt_cfg_scale_slider,
            guidance_scale_img    = img_cfg_scale_slider,
            width                 = width_slider,
            height                = height_slider,
            video_length          = 16,
            noise_sampling_method = "pyoco_mixed",
            noise_alpha           = 1.0,
            frame_stride          = frame_stride,
            use_frameinit         = use_frameinit,
            frameinit_noise_level = frame_init_noise_level,
            camera_motion         = None,
        ).videos

        global sample_idx
        sample_idx += 1
        save_sample_path = os.path.join(self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path, format="mp4")
    
        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "first_frame_path": input_image_path,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale_text": txt_cfg_scale_slider,
            "guidance_scale_image": img_cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": 8,
            "seed": seed
        }
        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        return gr.Video.update(value=save_sample_path)
        

controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # ConsistI2V Text+Image to Video Generation
            Input image will be used as the first frame of the video. Text prompts will be used to control the output video content.
            """
        )

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                - Input image can be specified using the "Input Image Path/URL" text box (this can be either a local image path or an image URL) or uploaded by clicking or dragging the image to the "Input Image" box. The uploaded image will be temporarily stored in the "samples/Gradio" folder under the project root folder.
                - Input image can be resized and/or center cropped to a given resolution by adjusting the "Width" and "Height" sliders. It is recommended to use the same resolution as the training resolution (256x256).
                - After setting the input image path or changed the width/height of the input image, press the "Preview" button to visualize the resized input image.
                """
            )
            
            with gr.Row():
                prompt_textbox = gr.Textbox(label="Prompt", lines=2)
                negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2)
                
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown   = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=50, minimum=10, maximum=250, step=1)
                    
                    with gr.Row():
                        center_crop   = gr.Checkbox(label="Center Crop the Image", value=True)
                        width_slider  = gr.Slider(label="Width",  value=256, minimum=0, maximum=512, step=64)
                        height_slider = gr.Slider(label="Height", value=256, minimum=0, maximum=512, step=64)
                    with gr.Row():
                        txt_cfg_scale_slider = gr.Slider(label="Text CFG Scale",   value=7.5, minimum=1.0,   maximum=20.0, step=0.5)
                        img_cfg_scale_slider = gr.Slider(label="Image CFG Scale",  value=1.0, minimum=1.0,   maximum=20.0, step=0.5)
                        frame_stride         = gr.Slider(label="Frame Stride",     value=3,   minimum=1,     maximum=5,    step=1)
                    
                    with gr.Row():
                        use_frameinit = gr.Checkbox(label="Enable FrameInit", value=True)
                        frameinit_noise_level = gr.Slider(label="FrameInit Noise Level", value=850, minimum=1, maximum=999, step=1)

                        
                        seed_textbox = gr.Textbox(label="Seed", value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])
                    
                    
            
                    generate_button = gr.Button(value="Generate", variant='primary')
                
                with gr.Column():
                    with gr.Row():
                        input_image_path = gr.Textbox(label="Input Image Path/URL", lines=1, scale=10, info="Press Enter or the Preview button to confirm the input image.")
                        preview_button = gr.Button(value="Preview")
                    
                    with gr.Row():
                        input_image = gr.Image(label="Input Image", interactive=True)
                        input_image.upload(fn=controller.update_textbox_and_save_image, inputs=[input_image, height_slider, width_slider, center_crop], outputs=[input_image_path, input_image])
                        result_video = gr.Video(label="Generated Animation", interactive=False, autoplay=True)

            def update_and_resize_image(input_image_path, height_slider, width_slider, center_crop):
                if input_image_path.startswith("http://") or input_image_path.startswith("https://"):
                    pil_image = Image.open(requests.get(input_image_path, stream=True).raw).convert('RGB')
                else:
                    pil_image = Image.open(input_image_path).convert('RGB')
                controller.image_resolution = pil_image.size
                original_width, original_height = pil_image.size
                
                if center_crop:
                    crop_aspect_ratio = width_slider / height_slider
                    aspect_ratio = original_width / original_height
                    if aspect_ratio > crop_aspect_ratio:
                        new_width = int(crop_aspect_ratio * original_height)
                        left = (original_width - new_width) / 2
                        top = 0
                        right = left + new_width
                        bottom = original_height
                        pil_image = pil_image.crop((left, top, right, bottom))
                    elif aspect_ratio < crop_aspect_ratio:
                        new_height = int(original_width / crop_aspect_ratio)
                        top = (original_height - new_height) / 2
                        left = 0
                        right = original_width
                        bottom = top + new_height
                        pil_image = pil_image.crop((left, top, right, bottom))
                
                pil_image = pil_image.resize((width_slider, height_slider))
                return gr.Image.update(value=np.array(pil_image))
            
            preview_button.click(fn=update_and_resize_image, inputs=[input_image_path, height_slider, width_slider, center_crop], outputs=[input_image])
            input_image_path.submit(fn=update_and_resize_image, inputs=[input_image_path, height_slider, width_slider, center_crop], outputs=[input_image])

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    prompt_textbox,
                    negative_prompt_textbox,
                    input_image_path,
                    sampler_dropdown,
                    sample_step_slider,
                    width_slider,
                    height_slider,
                    txt_cfg_scale_slider,
                    img_cfg_scale_slider,
                    center_crop,
                    frame_stride,
                    use_frameinit,
                    frameinit_noise_level,
                    seed_textbox,
                ],
                outputs=[result_video]
            )
            
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)
