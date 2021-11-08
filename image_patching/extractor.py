from typing import List, Optional
import torch


class PatchExtractor:
    """
    patch_size: int                             -   Extracted patches are rectangular with side lenght patch_size
    image_shape: Optional[List[int]] = None     -   Optionally provide image size (bs, c, h, w), otherwise it is determined on first call.
    resize: Optional[str] = 'truncate'          -   Determine what to do with non-square or mismatching images. [interpolate | truncate]
    """

    def __init__(self,
        patch_size: int,
        image_shape: Optional[List[int]] = None,
        resize: Optional[str] = 'truncate',
        ) -> None:

        self.ps = patch_size
        self.resize = resize
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
            if images.shape[-1] != images.shape[-2]:
                raise ValueError('Trying to determine image-size from non-rectangular image. Pass image_shape in to the constructor instead.')
            self._set_image_shape(images.shape)

        images = self._resize_images(images)

        # unfold and reshape to (np * bs, nc, ps, ps)

        unfolded = images.data.unfold(2, self.ps, self.ps).unfold(3, self.ps, self.ps)
        reordered = unfolded.reshape(-1, self.nc, self.np, self.ps, self.ps).permute(2, 0, 1, 3, 4)     # self.bs           -> -1
        patches = reordered.reshape(-1, self.nc, self.ps, self.ps)                                      # self.np * self.bs -> -1

        return patches

    def reconstruct_image(self, patches: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of images from a batch of image patches"""
        
        if not self._input_shape_set:
            raise ValueError("No input shape provided. Call split_to_patches() first.")

        assert patches.shape[1:] == (self.nc, self.ps, self.ps)

        # Currently, fold only supports 3D-tensors, shapes should be:
        # Input: (N, C × ∏(kernel_size), L) -> (bs, nc * ps * ps, np) 
        # Output: (N, C, output_shape[0], output_shape[1], ...) -> (bs, nc, h, w)
        
        reordered = patches.reshape(self.np, -1, self.nc * self.ps * self.ps).permute(1, 2, 0)          # self.bs           -> -1
        images = torch.nn.functional.fold(reordered, output_size=(self.size, self.size), kernel_size=self.ps, stride=self.ps)

        return images

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Resize image tensor to a suitable dimension."""

        if self.resize == 'truncate':
            side = min(images.shape[2:]) - min(images.shape[2:]) % self.ps
            images = images[:, :, :side, :side]
        elif self.resize == 'interpolate':
            side = min(images.shape[2:]) - min(images.shape[2:]) % self.ps
            images = torch.nn.functional.interpolate(images, (side, side))
        else:
            raise NotImplementedError(f"Resize method {self.resize} not implemented")
        return images