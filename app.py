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

#prompt_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
#prompt_pipe.to(device)

img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
img_pipe.to(device)

source_img = gr.Image(source="upload", type="filepath", label="init_img | 512*512 px")
gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

def resize(value,img):
  #baseheight = value
  img = Image.open(img)
  #hpercent = (baseheight/float(img.size[1]))
  #wsize = int((float(img.size[0])*float(hpercent)))
  #img = img.resize((wsize,baseheight), Image.Resampling.LANCZOS)
  img = img.resize((value,value), Image.Resampling.LANCZOS)
  return img


def infer(source_img, prompt): 
         
    source_image = resize(512, source_img)
    source_image.save('source.png')
    images_list = img_pipe([prompt] * 2, init_image=source_image, strength=0.75)
    images = []
    safe_image = Image.open(r"unsafe.png")
    for i, image in enumerate(images_list["sample"]):
        if(images_list["nsfw_content_detected"][i]):
            images.append(safe_image)
        else:
            images.append(image)    
    return images

print("Great sylvain ! Everything is working fine !")

title="Img2Img Stable Diffusion CPU"
description="Img2Img Stable Diffusion example using CPU and HF token. <br />Warning: Slow process... ~5/10 min inference time. <b>NSFW filter enabled.</b>" 

gr.Interface(fn=infer, inputs=[source_img, "text"], outputs=gallery,title=title,description=description).queue(max_size=100, allow_flagging="manual").launch(enable_queue=True)