"""
Loader for ImageNet. Borrowed heavily from
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import os
import torch.utils.data as data
from torchvision import transforms, datasets

IMAGENET_DIR = None
DIR_LIST = ['/data5/honglinc/Dataset/imagenet_raw',
            '/data5/chengxuz/Dataset/imagenet_raw',
            '/data5/chengxuz/imagenet_raw',
            '/data/chengxuz/imagenet_raw']

for path in DIR_LIST:
    if os.path.exists(path):
        IMAGENET_DIR = path
        break

assert IMAGENET_DIR is not None

class ImageNet(data.Dataset):
    def __init__(self, train=True, imagenet_dir=IMAGENET_DIR, image_transforms=None):
        super().__init__()
        split_dir = 'train' if train else 'validation'
        self.imagenet_dir = os.path.join(imagenet_dir, split_dir)
        self.dataset = datasets.ImageFolder(self.imagenet_dir, image_transforms)

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # important to return the index!
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)
