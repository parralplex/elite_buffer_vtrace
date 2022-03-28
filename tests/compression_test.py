import unittest
import torch

from utils.compression import compress, decompress


class TestCompression(unittest.TestCase):

    def test_tensor(self):
        rand_tensor = torch.rand(4, 80, 80, dtype=torch.float)
        compressed_tensor = compress(rand_tensor)
        decomp_tensor = decompress(compressed_tensor)
        self.assertTrue(torch.equal(rand_tensor, decomp_tensor), 'tensor after compression cycle is not the same')


