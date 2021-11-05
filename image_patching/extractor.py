from typing import List, Optional
import torch


class PatchExtractor:

    def __init__(self,
    patch_size: int,
    image_shape: Optional[List[int]] = None,
    ) -> None:
        self.ps = patch_size
        self._input_shape_set = False

        if not image_shape is None:
            self._set_image_shape(image_shape)

    def _set_image_shape(self, shape: list) -> None:
        """Set input shape"""
        # Assuming (bs, c, s, s), where s is the side of the images
        assert len(shape) == 4
        assert shape[-1] == shape[-2]
        self.bs = shape[0]
        self.nc = shape[1]  
        self.size = shape[-1]
        self.np = (self.size // self.ps) ** 2
        self._input_shape_set = True

    def split_to_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images into a batch of image patches."""

        if not self._input_shape_set:
            self._set_image_shape(images.shape)

        # unfold and reshape to (np * bs, nc, ps, ps)

        unfolded = images.data.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        reordered = unfolded.reshape(self.bs, self.nc, self.np, self.ps, self.ps).permute(2, 0, 1, 3, 4)
        patches = reordered.reshape(self.np * self.bs, self.nc, self.ps, self.ps)

        return patches

    def reconstruct_image(self, patches: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of images from a batch of image patches"""
        
        if not self._input_shape_set:
            raise ValueError("No input shape provided. Call split_to_patches() first.")

        # Currently, fold only supports 3D-tensors, shapes should be:
        # Input: (N, C × ∏(kernel_size), L) -> (bs, nc * ps * ps, np) 
        # Output: (N, C, output_shape[0], output_shape[1], ...) -> (bs, nc, h, w)
        
        reordered = patches.reshape(self.np, self.bs, self.nc * self.ps * self.ps).permute(1, 2, 0)
        images = torch.nn.functional.fold(reordered, output_size=(self.size, self.size), kernel_size=self.ps, stride=self.ps)

        return images

