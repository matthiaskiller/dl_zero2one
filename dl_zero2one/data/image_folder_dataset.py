"""
Definition of ImageFolderDataset dataset class
"""

import os
import random

import numpy as np
from PIL import Image

from .base_dataset import Dataset


class ImageFolderDataset(Dataset):
    """CIFAR-10 dataset class"""
    def __init__(self, transform=None, mode='train', limit_files=None,
                 split={'train': 0.6, 'val': 0.2, 'test': 0.2},
                 download_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                 *args, **kwargs):
        super().__init__(*args, 
                         download_url=download_url,
                         **kwargs)

        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0
        
        self.classes, self.class_to_idx = self._find_classes(self.root_path)
        self.split = split
        self.limit_files = limit_files
        self.images, self.labels = self.make_dataset(
            directory=self.root_path,
            class_to_idx=self.class_to_idx,
            mode=mode,
        )
        self.transform = transform

    def make_dataset(self, directory, class_to_idx, mode):
        """
        Create the image dataset by preparaing a list of samples
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset
            - labels is a list containing one label per image
        """
        images, labels = [], []
        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    images.append(path)
                    labels.append(label)

        images, labels = self.select_split(images, labels, mode)

        assert len(images) == len(labels)
        return images, labels
    
    
    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def select_split(self, images, labels, mode):
        """
        Depending on the mode of the dataset, deterministically split it.
        
        :param images, a list containing paths to all images in the dataset
        :param labels, a list containing one label per image
        
        :returns (images, labels), where only the indices for the
            corresponding data split are selected.
        """
        fraction_train = self.split['train']
        fraction_val = self.split['val']
        num_samples = len(images)
        num_train = int(num_samples * fraction_train)
        num_valid = int(num_samples * fraction_val)
        
        np.random.seed(0)
        rand_perm = np.random.permutation(num_samples)
        
        if mode == 'train':
            idx = rand_perm[:num_train]
        elif mode == 'val':
            idx = rand_perm[num_train:num_train+num_valid]
        elif mode == 'test':
            idx = rand_perm[num_train+num_valid:]

        if self.limit_files:
            idx = idx[:self.limit_files]
            
        return list(np.array(images)[idx]), list(np.array(labels)[idx])

    
        
    def __len__(self):
        length = None
        length = len(self.images)
        return length

    @staticmethod
    def load_image_as_numpy(image_path):
        """Load image from image_path as numpy array"""
        return np.asarray(Image.open(image_path), dtype=float)

    def __getitem__(self, index):
        data_dict = None

        label = self.labels[index]
        path = self.images[index]
        image = self.load_image_as_numpy(path)
        if self.transform is not None:
            image = self.transform(image)
        data_dict = {
            "image": image,
            "label": label,
        }

        return data_dict

class RandomHorizontalFlip:
    """
    Transform class that flips an image horizontically randomly with a given probability.
    """

    def __init__(self, prob=0.5):
        """
        :param prob: Probability of the image being flipped
        """
        self.p = prob

    def __call__(self, image):
        rand = random.uniform(0,1)
        if rand < self.p:
            image = np.flip(image,1)
        return image
