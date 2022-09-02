pip install -U git+https://github.com/huggingface/diffusers
import gradio as gr
#import torch
#from torch import autocast // only for GPU

from PIL import Image

import os
MY_SECRET_TOKEN=os.environ.get('HF_TOKEN_SD')

from diffusers import StableDiffusionPipeline
#from diffusers import StableDiffusionImg2ImgPipeline

print("hello sylvain")

YOUR_TOKEN=MY_SECRET_TOKEN

device="cpu"

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe.to(device)

gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

def infer(prompt): 
    
    #image = pipe(prompt, init_image=init_image)["sample"][0]
    images_list = pipe([prompt] * 4)
    images = []
    safe_image = Image.open(r"unsafe.png")
    for i, image in enumerate(images_list["sample"]):
        if(images_list["nsfw_content_detected"][i]):
            images.append(safe_image)
        else:
            images.append(image)
    
    return images

print("Great sylvain ! Everything is working fine !")

title="Stable Diffusion CPU"
description="Stable Diffusion example using CPU and HF token. <br />Warning: Slow process... ~5/10 min inference time. <b>NSFW filter enabled.</b>" 

gr.Interface(fn=infer, inputs="text", outputs=gallery,title=title,description=description).launch(enable_queue=True)