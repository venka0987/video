import gradio as gr
#import torch
#from torch import autocast // only for GPU

from PIL import Image
import numpy as np
from io import BytesIO
import os
MY_SECRET_TOKEN=os.environ.get('HF_TOKEN_SD')

#from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

print("hello sylvain")

YOUR_TOKEN=MY_SECRET_TOKEN

device="cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)

source_img = gr.Image(source="upload", type="numpy")
gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

def infer(prompt, init_image): 
    print(init_image)
    return init_image
print("Great sylvain ! Everything is working fine !")

title="Stable Diffusion CPU"
description="Stable Diffusion example using CPU and HF token. <br />Warning: Slow process... ~5/10 min inference time. <b>NSFW filter enabled.</b>" 

gr.Interface(fn=infer, inputs=["text", source_img], outputs=gallery,title=title,description=description).launch(enable_queue=True)