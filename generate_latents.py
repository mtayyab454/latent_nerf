from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

image_folder = "./data/nerfstudio/fangzhou-small/images"
latents_folder = "./data/nerfstudio/fangzhou-small/latents"

# create latents folder if it doesn't exist
if not os.path.exists(latents_folder):
    os.mkdir(latents_folder)

# Load the model
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae = vae.to("cuda")
vae = vae.half()

# torch transform, to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# load image names from folder
image_names = os.listdir(image_folder)
# keep average of min and max latents
min_latents = []
max_latents = []

for i in range(len(image_names)):
    # read image
    image_name = image_names[i]
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path)

    image = transform(image).unsqueeze(0).half().to("cuda")

    encoder_out = vae.encode(image)
    latents = encoder_out['latent_dist'].mean
    latents = latents.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

    # keep track of min and max latents
    min_latents.append(latents.min())
    max_latents.append(latents.max())

    # delete variables to free up memory

    del latents
    del image
    del encoder_out

    # print(torch.cuda.memory_summary(device=None, abbreviated=False))


    # normalize latents to [0, 1]
    # latents = (latents - latents.min()) / (latents.max() - latents.min())

    # latents = latents.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    #
    # latents = torch.from_numpy(latents).to("cuda")
    # latents = latents.unsqueeze(0).permute(0, 3, 1, 2)
    # im = pipe.vae.decode(latents).sample
    #
    # # normalize im and save as png image
    # im = im.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    # im = (im - im.min()) / (im.max() - im.min())
    # latent_image = Image.fromarray((im * 255).astype(np.uint8))
    # latent_image.show()

    # latent_image = Image.fromarray((latents * 255).astype(np.uint8))
    # latent_image.save(latent_path.replace(".npy", ".png"))

# display average of min and max latents
print(len(min_latents))
print(f"min min: {np.min(min_latents)}, max max: {np.max(max_latents)}, max range: {np.max(max_latents) - np.min(min_latents)}")
print(f"average min: {np.mean(min_latents)}, average max: {np.mean(max_latents)}, average range: {np.mean(max_latents) - np.mean(min_latents)}")