# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import Tensor, nn
from jaxtyping import Float

CONSOLE = Console(width=120)

try:
    from diffusers import (
        DDIMScheduler,
        StableDiffusionInstructPix2PixPipeline,
        AutoencoderKL,
    )
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"

from diffusers import UNet2DModel

import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from ln.ln_adapter import MyAdapter

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor


class RefinementModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.refinment_model = MyAdapter()
        self.refinment_model = self.refinment_model.half().to("cuda")

    def train_refinement(self, gt_latents_folder, target_latents_folder, training_steps=1):
        print("Training refinement model")
        self.refinment_model.train()
        # latents_folder = data_folder / "latents"

        # optimizer = torch.optim.Adam(self.refinment_model.parameters(), lr=0.0001)
        optimizer = torch.optim.SGD(self.refinment_model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.MSELoss()

        # torch transform, to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            # transforms.Resize((64, 64)),
        ])

        # list all png images in latents folder
        file_names = os.listdir(target_latents_folder)
        image_names = [i for i in file_names if i.endswith(".png") and not i.endswith("_decoded.png")]
        # create list of npy names from image names by replacing png with npy
        npy_names = [i.replace(".png", ".npy") for i in image_names]

        for j in range(training_steps):
            loss_vec = []
            for i in range(len(image_names)):
                # read image
                # print(f"Generating latent {i + 1}/{len(image_names)}")
                image_name = image_names[i]
                image_path = os.path.join(target_latents_folder, image_name)
                image = Image.open(image_path)
                image = transform(image).unsqueeze(0).half().to("cuda")

                # read latent
                latent_name = npy_names[i]
                latent_path = os.path.join(gt_latents_folder, latent_name)
                latent = np.load(latent_path)
                latent = torch.from_numpy(latent).unsqueeze(0).permute(0, 3, 1, 2).half().to("cuda")

                # resize latent to 64x64
                # latent = torch.nn.functional.interpolate(latent, size=(64, 64), mode='bilinear')

                # train refinement model
                optimizer.zero_grad()
                output = self.refinment_model(image)
                loss = criterion(output, latent)
                loss.backward()
                optimizer.step()

                loss_vec.append(loss.item())

            # print loss and train step
            print(f"Step {j+1}/{training_steps}, loss: {np.mean(loss_vec)}")
            # print model parameters
            # print(f"range {self.refinment_model.range.item()}, min: {self.refinment_model.min.item()}")
#
# # number of parameters in the model
# print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")

