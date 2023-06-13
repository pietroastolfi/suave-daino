"""
3Augment Data-augmentation (DA) based on DeiT III (https://github.com/facebookresearch/deit)
"""
from torchvision import transforms
import random

from PIL import ImageFilter, ImageOps


class SimpleResizedCrop(object):
    """
    Apply Simple Resized crop.
    """
    def __init__(self, img_size=224):

        self.t = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
        ])

    def __call__(self, img):
        return self.t(img)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Augment3(object):
    """
    Apply 3Augment as in deit III 
    """
    def __init__(self, img_size, data_mean, data_std, simple_crop=False, jitter=0.3, n_rep=1):
        
        self.n_rep = n_rep
        t0 = SimpleResizedCrop(img_size) if simple_crop else transforms.RandomResizedCrop(img_size)
        t1 = transforms.RandomHorizontalFlip()
        t2 = transforms.RandomChoice([
            transforms.RandomGrayscale(p=1.0),
            Solarization(p=1.0),
            GaussianBlur(p=1.0)
        ])
        t3 = transforms.ColorJitter(jitter, jitter, jitter)
        t4 = transforms.ToTensor()
        t5 = transforms.Normalize(mean=data_mean, std=data_std)

        self.t = transforms.Compose([t0, t1, t2, t3, t4, t5])

    def __call__(self, img):
        return [self.t(img) for _ in range(self.n_rep)]
