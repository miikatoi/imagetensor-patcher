import torch

from image_patching import PatchExtractor

patch_extractor = PatchExtractor(patch_size=128)
batch = torch.Tensor(32, 3, 512, 512)

# split image into a grid of equally sized patches
patches_batch = patch_extractor.split_to_patches(batch)

# reconstruct patches into the original size
reconstructed_images_batch = patch_extractor.reconstruct_image(patches_batch)

print(batch.shape)
print(patches_batch.shape)
print(reconstructed_images_batch.shape)
