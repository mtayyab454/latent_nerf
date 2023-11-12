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

@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor


class RefinementModel(nn.Module):

    def __init__(self, image_size) -> None:
        super().__init__()

        self.refinment_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=4, out_channels=4, init_features=32, pretrained=False)
        self.refinment_model = self.refinment_model.half().to("cuda")

    def train_refinement(self, data_folder, training_steps=1000):
        print("Training refinement model")
        self.refinment_model.train()
        latents_folder = data_folder / "latents"

        optimizer = torch.optim.Adam(self.refinment_model.parameters(), lr=0.00001)
        criterion = torch.nn.MSELoss()

        # torch transform, to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
            transforms.Resize((64, 64)),
        ])

        # list all png images in latents folder
        file_names = os.listdir(latents_folder)
        image_names = [i for i in file_names if i.endswith(".png")]
        npy_names = [i for i in file_names if i.endswith(".npy")]

        for i in range(training_steps):
            for i in range(len(image_names)):
                # read image
                # print(f"Generating latent {i + 1}/{len(image_names)}")
                image_name = image_names[i]
                image_path = os.path.join(latents_folder, image_name)
                image = Image.open(image_path)
                image = transform(image).unsqueeze(0).half().to("cuda")

                # read latent
                latent_name = npy_names[i]
                latent_path = os.path.join(latents_folder, latent_name)
                latent = np.load(latent_path)
                latent = torch.from_numpy(latent).unsqueeze(0).permute(0, 3, 1, 2).half().to("cuda")

                # resize latent to 64x64
                latent = torch.nn.functional.interpolate(latent, size=(64, 64), mode='bilinear')

                # train refinement model
                optimizer.zero_grad()
                output = self.refinment_model(image)
                loss = criterion(output, latent)
                loss.backward()
                optimizer.step()

                # print loss and train step
                print(f"Step {i+1}/{training_steps}, loss: {loss.item()}")

class FullyConvNet(nn.Module):
    def __init__(self):
        super(FullyConvNet, self).__init__()
        # Define convolutional layers
        # Layer 1 (input layer with 4 channels)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, 16)
        self.relu1 = nn.ReLU()

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, 32)
        self.relu2 = nn.ReLU()

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.GroupNorm(8, 64)
        self.relu3 = nn.ReLU()

        # Output layer (output with 4 channels)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1)

    def forward(self, x):
        # Forward pass through each layer
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # Output layer
        return x
