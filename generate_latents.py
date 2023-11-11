from diffusers import StableDiffusionPipeline
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
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# torch transform, to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# load image names from folder
image_names = os.listdir(image_folder)

for i in range(1):
    # read image
    image_name = image_names[i]
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path)

    image = transform(image).unsqueeze(0).half().to("cuda")

    encoder_out = pipe.vae.encode(image)
    latents = encoder_out['latent_dist'].mean

    # display min and range of latents in a single line
    print(f"min: {latents.min()}, max: {latents.max()}, range: {latents.max() - latents.min()}")


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






