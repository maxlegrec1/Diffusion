import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F
from torchvision import transforms


class Transform(torch.nn.Module):
    def __init__(self,img_size = 256):
        super().__init__()
        self.transform_global = transforms.Compose(
            [   v2.ElasticTransform(alpha = random.uniform(0, 50), sigma = random.uniform(3, 7)),
                v2.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                v2.RandomRotation(
                    10
                ),  # Randomly rotate the image by up to 10 degrees
                v2.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                v2.Resize((img_size,img_size))
            ]
        )
    def forward(self, x):
        


        x = self.transform_global(x)

        return x


class Dataset:

    def __init__(self, dataset_path, dir_of_dirs=True, extension="jpg",size = 256):
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
        self.info()
        self.transform = Transform(size)
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
        self, batch_size=32, device="cuda", transform=None, loop_when_finish=True
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
        if transform == None:
            transform = self.transform
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
                            final_batch = self.normalize_images(final_batch)
                            yield final_batch
                        return

                # Read image
                image = torchvision.io.read_image(self.images_path[perm[current_idx]])

                # Add to batch
                batch.append(transform(image))

                current_idx += 1

            # Convert batch to tensor
            batch = torch.stack(batch).to(device)
            # Apply transform if provided

            # Normalize the batch
            batch = self.normalize_images(batch)

            yield batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset_path = "datasets/afhq/train"

    ds = Dataset(dataset_path,size = 128)
    ds.info()
    gen = ds.create_gen(batch_size=16, loop_when_finish=False)
    num_img_show = 10
    for i in range(num_img_show):
        batch = next(gen)
        img = (batch[0].permute(1,2,0).cpu().numpy() + 1 )/2
        plt.imshow(img)
        plt.show()
    exit()