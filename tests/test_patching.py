import unittest
import torch

from image_patching.extractor import PatchExtractor


class TestPatchExtractor(unittest.TestCase):

    def setUp(self):
        self.bs, self.nc, self.s = 2, 3, 256
        self.x = torch.rand((self.bs, self.nc, self.s, self.s))
        self.ps = 64
        self.extractor = PatchExtractor(patch_size=self.ps)

        self.np = (self.s // self.ps) ** 2

    def test_patch_extraction(self):
        patches = self.extractor.split_to_patches(self.x)
        self.assertEqual(patches.shape, (self.bs * self.np, self.nc, self.ps, self.ps))

    def test_reconstruction(self):
        patches = self.extractor.split_to_patches(self.x)
        reconstruction = self.extractor.reconstruct_image(patches)
        self.assertEqual(reconstruction.shape, (self.bs, self.nc, self.s, self.s))
    
    def test_equality(self):
        reconstruction = self.extractor.reconstruct_image(self.extractor.split_to_patches(self.x))
        self.assertTrue((self.x == reconstruction).all())


if __name__ == '__main__':

    unittest.main()


