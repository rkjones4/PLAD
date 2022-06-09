# This contains data generator function for shapenet rendered images
import h5py
import numpy as np


class Generator:
    def __init__(self):
        pass

    def train_gen(
            self,
            batch_size,
            path="data/shapenet/shuffled_images_splits.h5",
    ):
        with h5py.File(path, "r") as hf:
            images = np.array(hf.get(name="train_images"))

        while True:
            for i in range(images.shape[0] // batch_size):
                mini_batch = images[batch_size * i:batch_size * (i + 1)]
                mini_batch = np.expand_dims(mini_batch, 1)

                yield np.expand_dims(mini_batch, 0).astype(np.float32)

    def val_gen(self,
                batch_size,
                path="data/shapenet/shuffled_images_splits.h5",
    ):
        with h5py.File(path, "r") as hf:
            images = np.array(hf.get("val_images"))
        while True:
            for i in range(images.shape[0] // batch_size):
                mini_batch = images[batch_size * i:batch_size * (i + 1)]
                mini_batch = np.expand_dims(mini_batch, 1)

                yield np.expand_dims(mini_batch, 0).astype(np.float32)

    def test_gen(self,
                 batch_size,
                 path="data/shapenet/shuffled_images_splits.h5",
                 ):
        with h5py.File(path, "r") as hf:
            images = np.array(hf.get("test_images"))

        for i in range(images.shape[0] // batch_size):
            mini_batch = images[batch_size * i:batch_size * (i + 1)]
            mini_batch = np.expand_dims(mini_batch, 1)

            yield np.expand_dims(mini_batch, 0).astype(np.float32)
