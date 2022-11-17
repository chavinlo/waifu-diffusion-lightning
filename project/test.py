import torch
import time
import argparse

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

parser = argparse.ArgumentParser(description="Try out diffusers models")
parser.add_argument('-m', '--model', type=str, required=True, help="Path to the model folder in diffuser format")
parser.add_argument('-p', '--promptfile', type=str, required=True, help="Path to the text file with prompt to be used")
parser.add_argument('-o', '--outputfolder', type=str, required=False, help="Path to the folder to output outputs")
args = parser.parse_args()

folder_path = args.model

device = torch.device('cuda')

tokenizer = CLIPTokenizer.from_pretrained(folder_path, subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(folder_path, subfolder='text_encoder')
vae = AutoencoderKL.from_pretrained(folder_path, subfolder='vae')
unet = UNet2DConditionModel.from_pretrained(folder_path, subfolder='unet')

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

vae = vae.to(device, dtype=torch.float32)
unet = unet.to(device, dtype=torch.float32)
text_encoder = text_encoder.to(device, dtype=torch.float32)

print('using DDIMScheduler scheduler')
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)

pipeline = StableDiffusionPipeline(
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    tokenizer=tokenizer,
    scheduler=scheduler,
    safety_checker=None, # disable safety checker to save memory
    feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
).to(device)

saveInferencePath = args.outputfolder

prompt = open(args.promptfile, "r").read

for _ in range(10):
    from datetime import datetime
    print("OK")
    images = pipeline(prompt, num_inference_steps=50).images[0]
    filenameImg = str(time.time_ns()) + ".png"
    filenameTxt = str(time.time_ns()) + ".txt"
    images.save(saveInferencePath + "/" + filenameImg)
    with open(saveInferencePath + "/" + filenameTxt, 'a') as f:
        f.write('Used prompt: ' + prompt + '\n')
        f.write('Generated Image Filename: ' + filenameImg + '\n')
        f.write('Generated at: ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+ '\n')
