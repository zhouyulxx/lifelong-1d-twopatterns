import unittest
import numpy as np

from continual_1d.data import normalize_per_sample, get_task_indices
from continual_1d.model import SequenceMLP

class CoreTests(unittest.TestCase):
    def test_normalization(self):
        x = np.vstack([np.arange(128), np.arange(128) * 2]).astype(np.float32)
        z = normalize_per_sample(x)
        self.assertEqual(z.shape, (2, 128))
        self.assertTrue(np.allclose(z.mean(axis=1), 0, atol=1e-5))
        self.assertTrue(np.allclose(z.std(axis=1), 1, atol=1e-5))

    def test_disjoint_tasks(self):
        y = np.asarray([0, 1, 2, 3, 0, 2])
        a = get_task_indices(y, (0, 1))
        b = get_task_indices(y, (2, 3))
        self.assertEqual(len(set(a) & set(b)), 0)
        self.assertEqual(len(a) + len(b), len(y))

    def test_model_shapes(self):
        import torch
        model = SequenceMLP()
        self.assertEqual(tuple(model(torch.zeros((7, 128))).shape), (7, 4))

if __name__ == "__main__":
    unittest.main()
