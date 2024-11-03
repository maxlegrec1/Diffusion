import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms


class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        rotation_angle = 20 * random.random() - 10
        x = transforms.functional.rotate(x, rotation_angle)

        flip = random.random() >= 0.5

        if flip:
            x = transforms.functional.hflip(x)

        alpha = random.uniform(0, 50)  # Controls the intensity of the transform
        sigma = random.uniform(3, 7)  # Controls the smoothness of the transform
        sigma = [sigma, sigma]
        alpha = [alpha, alpha]
        size = list(x.shape[-2:])
        dx = torch.rand([1, 1] + size) * 2 - 1
        if sigma[0] > 0.0:
            kx = int(8 * sigma[0] + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = F.gaussian_blur(dx, [kx, kx], sigma)
        dx = dx * alpha[0] / size[0]

        dy = torch.rand([1, 1] + size) * 2 - 1
        if sigma[1] > 0.0:
            ky = int(8 * sigma[1] + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = F.gaussian_blur(dy, [ky, ky], sigma)
        dy = dy * alpha[1] / size[1]
        displacement = torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

        x = F.elastic_transform(x, displacement=displacement, fill=0)
        transform_global = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                transforms.RandomRotation(
                    10
                ),  # Randomly rotate the image by up to 10 degrees
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
            ]
        )
        x = transform_global(x)

        return x


class Dataset:

    def __init__(self, dataset_path, dir_of_dirs=True, extension="png"):
        self.path = dataset_path
        if dir_of_dirs:
            self.images_path = []
            for d in os.listdir(self.path):
                if os.path.isdir(os.path.join(self.path, d)):
                    for img in os.listdir(os.path.join(self.path, d)):
                        if img.endswith(f".{extension}"):
                            self.images_path.append(os.path.join(self.path, d, img))
        else:
            self.images_path = [
                os.path.join(self.path, image_path)
                for image_path in os.listdir(self.path)
            ]

        self.img_size = torchvision.io.read_image(self.images_path[0]).shape[1:]

    def info(self):

        print(f"Dataset length : {len(self.images_path)}")

        print(f"Image Size : {self.img_size}")

    def get_one_image(self):
        image = torchvision.io.read_image(self.images_path[0])
        print(f" Returning one image of size {self.img_size} and of type {image.dtype}")
        image = self.normalize_images(image)
        return image

    def normalize_images(self, batch):
        # normalize between -1 and 1
        return (batch - 127.5) / 127.5

    def create_gen(
        self, batch_size=32, device="cuda", transform=Transform(), loop_when_finish=True
    ):
        """
        Creates a generator that yields batches of images.

        Args:
            batch_size (int): Number of images per batch
            device (str): Device to load images to ('cuda' or 'cpu')
            transform (callable, optional): Optional transform to be applied on the images
            loop_when_finish (bool): Whether to restart from beginning when all images are processed

        Yields:
            torch.Tensor: Batch of images of shape [batch_size, channels, height, width]
        """
        current_idx = 0
        num_images = len(self.images_path)
        perm = np.random.permutation(num_images)

        while True:
            # Initialize empty batch
            batch = []

            # Fill the batch
            for _ in range(batch_size):
                # Check if we've reached the end
                if current_idx >= num_images:
                    if loop_when_finish:
                        current_idx = 0  # Reset to beginning
                    else:
                        # If batch is not empty, yield it before stopping
                        if batch:
                            final_batch = torch.stack(batch).to(device)
                            if transform:
                                final_batch = transform(final_batch)
                            final_batch = self.normalize_images(final_batch)
                            yield final_batch
                        return

                # Read image
                image = torchvision.io.read_image(self.images_path[perm[current_idx]])

                # Add to batch
                batch.append(image)

                current_idx += 1

            # Convert batch to tensor
            batch = torch.stack(batch).to(device)

            # Apply transform if provided
            if transform:
                batch = transform(batch)

            # Normalize the batch
            batch = self.normalize_images(batch)

            yield batch


if __name__ == "__main__":

    dataset_path = "datasets/afhq"

    ds = Dataset(dataset_path)
    ds.info()
    gen = ds.create_gen(batch_size=2, loop_when_finish=True)

    i = 0
    for batch in gen:
        print(i)
        i += 1

    print(i)
