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

"""InstructPix2Pix Pipeline and trainer"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
from nerfstudio.utils.io import load_from_json

from diffusers import AutoencoderKL
from ln.ln_refinement import RefinementModel

@dataclass
class LatentNerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""
    _target: Type = field(default_factory=lambda: LatentNerfPipeline)

    latent_scale: float = 0.125
    """Latent scale factor."""
    # latent_size is touple of (h, w)
    latent_size: tuple = (64, 40)
    """Latent scale factor."""

class LatentNerfPipeline(VanillaPipeline):
    """LatentNerf pipeline"""

    config: LatentNerfPipelineConfig

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        # check if latents.json exists
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        vae = vae.to("cuda").half()
        if not os.path.exists(config.datamanager.data / "latents.json"):
            self._modify_json(config.datamanager.data, config.latent_scale)
            self._generate_latents(config.datamanager.data, vae)

        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        self.vae = vae
        self.refinement_model = RefinementModel(self.config.latent_size).to("cuda")
        self.refinement_model.train_refinement(self.datamanager.dataparser.config.data)

    def save_renderings(self, step: int, base_dir, use_decoder=True):
        """Save renderings of the current model."""
        os.mkdir(base_dir / "renderings" / str(step))

        for current_spot in range(len(self.datamanager.train_dataset)):

            print("Rendering image", current_spot)
            # get original image from dataset
            # original_image = self.pipeline.datamanager.original_image_batch["image"][current_spot].to(self.device)
            # generate current index in datamanger
            current_index = self.datamanager.image_batch["image_idx"][current_spot]

            # get current camera, include camera transforms from original optimizer
            # camera_transforms = self.model.camera_optimizer(current_index.unsqueeze(dim=0))
            current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1))

            # get current render of nerf
            # original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
            camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
            rendered_latents = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)

            # save image as png
            rendered_image = rendered_latents[0].cpu().numpy()
            rendered_image = rendered_image.transpose(1, 2, 0)
            rendered_image = (rendered_image * 255).astype('uint8')
            im = Image.fromarray(rendered_image)
            im.save(base_dir / "renderings" / str(step) / f"{current_index}.png")

            if use_decoder:
                # decode latents
                refinded_latents = self.refinement_model.refinment_model(rendered_latents)
                refinded_latents = refinded_latents.half().to("cuda")
                im = self.vae.decode(refinded_latents).sample

                # normalize im and save as png image
                im = im.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
                im = (im - im.min()) / (im.max() - im.min())
                im = Image.fromarray((im * 255).astype(np.uint8))
                im.save(base_dir / "renderings" / str(step) / f"{current_index}_decoded.png")

    def _modify_json(self, data_path, latent_scale):
        # Load the JSON data from the file
        data = load_from_json(data_path / "transforms.json")

        # Apply the scale factor
        for key in ['fl_x', 'fl_y', 'cx', 'cy', 'h', 'w']:
            if key in data:
                data[key] *= latent_scale
            # save h and w as int
            if key in ['h', 'w']:
                data[key] = int(data[key])

        # Modify the file_path in frames
        for frame in data['frames']:
            frame['file_path'] = frame['file_path'].replace('images', 'latents')

        # Save the modified data into a new JSON file
        with open(data_path / "latents.json", 'w') as file:
            json.dump(data, file, indent=4)

    def _generate_latents(self, data_path, vae):
        print("Generating latents...")
        image_folder = data_path / "images"
        latents_folder = data_path / "latents"

        # create latents folder if it doesn't exist
        if not os.path.exists(latents_folder):
            os.mkdir(latents_folder)

        # torch transform, to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # load image names from folder
        image_names = os.listdir(image_folder)

        for i in range(len(image_names)):
            # read image
            print(f"Generating latent {i+1}/{len(image_names)}")
            image_name = image_names[i]
            image_path = os.path.join(image_folder, image_name)
            image = Image.open(image_path)

            image = transform(image).unsqueeze(0).half().to("cuda")

            encoder_out = vae.encode(image)
            latents = encoder_out['latent_dist'].mean.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()

            # save latents as numpy array
            latent_path = os.path.join(latents_folder, image_name.replace(".png", ".npy"))
            np.save(latent_path, latents)

            # normalize latents to [0, 1]
            latents = (latents - latents.min()) / (latents.max() - latents.min())

            # save latents as png image
            latent_image = Image.fromarray((latents * 255).astype(np.uint8))
            latent_image.save(latent_path.replace(".npy", ".png"))

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
