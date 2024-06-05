"""
Definition of image-specific transform classes
"""

import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""

    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        Args:
            out_range: Value range to which images should be rescaled to
            in_range: Old value range of the images
                e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images):
        """
        Args:
            images: numpy array of shape NxHxWxC
                (for N images with C channels of spatial size HxW)
        Returns:
            images: rescaled images to the given range
        """

        images = self.min + ((images - self._data_min) *
                             (self.max - self.min)) / (self._data_max - self._data_min)

        return images


def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    Args:
        images: numpy array of shape NxHxWxC
            (for N images with C channels of spatial size HxW)
    Returns:
        per-channels mean and std; numpy array of shape C
    """
    mean, std = None, None

    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    return mean, std


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """

    def __init__(self, mean, std):
        """
        Args:
            mean: mean of images to be normalized
                can be a single value, or a numpy array of size C
            std: standard deviation of images to be normalized
                can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        """
        Normalize the given images using the mean and std
        Args:
            images: numpy array of shape NxHxWxC
                (for N images with C channels of spatial size HxW)
        Returns:
            images: normalized images using the mean and std
        """
        
        images = (images - self.mean)/self.std

        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""

    def __init__(self, transforms):
        """
        Args:
            transforms: transforms to be combined
        """
       
        self.transforms = transforms

    def __call__(self, images):
        """
        Apply all transforms to the images
        Args:
            images: numpy array of shape NxHxWxC
                (for N images with C channels of spatial size HxW)
        Returns:
            images: transformed images
        """
        for transform in self.transforms:
            images = transform(images)
        return images

class FlattenTransform:
    """Transform class that reshapes an image into a 1D array"""

    def __call__(self, image):
        return image.flatten()
