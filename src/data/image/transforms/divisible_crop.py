# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Union
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


class DivisibleCrop:
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)

        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        cropped_height = height - (height % self.height_factor)
        cropped_width = width - (width % self.width_factor)

        image = TVF.center_crop(img=image, output_size=(cropped_height, cropped_width))
        return image


class DivisiblePad:
    """Legacy fallback — kept for backward compatibility.
    Prefer ResizeDivisible which combines resize + divisibility in one step.
    """
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)
        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        new_height = height + (self.height_factor - (height % self.height_factor)) % self.height_factor
        new_width = width + (self.width_factor - (width % self.width_factor)) % self.width_factor

        if new_height == height and new_width == width:
            return image

        antialias = True
        if isinstance(image, torch.Tensor):
            image = TVF.resize(image, [new_height, new_width],
                               interpolation=InterpolationMode.BILINEAR,
                               antialias=antialias)
        elif isinstance(image, Image.Image):
            image = image.resize((new_width, new_height), Image.BILINEAR)
        return image
