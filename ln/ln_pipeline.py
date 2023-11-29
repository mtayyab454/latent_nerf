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
        config: LatentNerfPipelineConfig,
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
        else:
            # load min and max
            with open(config.datamanager.data / "latents" / "min_max.txt", 'r') as file:
                self.latent_min = float(file.readline())
                self.latent_max = float(file.readline())

        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        self.vae = vae

    def save_renderings(self, step: int, base_dir, use_decoder=True):
        """Save renderings of the current model."""
        os.mkdir(base_dir / "renderings" / str(step))

        for i in range(len(self.datamanager.train_dataset.image_filenames)):

            print("Rendering image", i)
            current_index = self.datamanager.image_batch["image_idx"][i]
            current_fname = self.datamanager.train_dataset.image_filenames[current_index].name

            # ############################################################################################################
            # im = self.datamanager.image_batch['image'][i].numpy()
            # # load latent
            # latent_path = 'data/nerfstudio/fangzhou-small/latents/' + current_fname.replace(".png", ".npy")
            # latents = np.load(latent_path)
            # # normalize latents
            # latents = (latents + 57.40625) / (44.5625 + 57.40625)
            #
            # diff = np.abs(im - latents)
            # print("diff", diff.max())
            #
            # ############################################################################################################

            current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1))

            camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
            rendered_latents = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)

            # save image as png
            rendered_image = rendered_latents[0].cpu().numpy()
            rendered_image = rendered_image.transpose(1, 2, 0)
            rendered_image = (rendered_image * 255).astype('uint8')
            im = Image.fromarray(rendered_image)
            im.save(base_dir / "renderings" / str(step) / current_fname)

            if use_decoder:
                # decode latents
                # refinded_latents = self.refinement_model.refinment_model(rendered_latents.half())
                # refinded_latents = refinded_latents.half().to("cuda")

                # normalize latents
                # rendered_latents = self.datamanager.image_batch['image'][i].permute(2, 0, 1).unsqueeze(0).to("cuda").half()
                rendered_latents = rendered_latents * (self.latent_max - self.latent_min) + self.latent_min

                im = self.vae.decode(rendered_latents.half()).sample

                # normalize im and save as png image
                im = im.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
                im = (im - im.min()) / (im.max() - im.min())
                im = Image.fromarray((im * 255).astype(np.uint8))
                im.save(base_dir / "renderings" / str(step) / current_fname.replace(".png", "_decoded.png") )

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

        latent_min = float("inf")
        latent_max = float("-inf")

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

            # update min and max
            latent_min = min(latent_min, latents.min())
            latent_max = max(latent_max, latents.max())

        for i in range(len(image_names)):
            image_name = image_names[i]
            latent_path = os.path.join(latents_folder, image_name.replace(".png", ".npy"))
            latents = np.load(latent_path)

            # normalize latents
            latents = (latents - latent_min) / (latent_max - latent_min)

            # save latents as png image
            latent_path = os.path.join(latents_folder, image_name.replace(".png", ".png"))
            im = Image.fromarray((latents * 255).astype(np.uint8))
            im.save(latent_path, alpha=True)

        # save min and max
        with open(latents_folder / "min_max.txt", 'w') as file:
            file.write(f"{latent_min}\n{latent_max}")

        self.latent_min = latent_min
        self.latent_max = latent_max

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
