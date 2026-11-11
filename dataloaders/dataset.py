import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
from PIL import Image
import sys

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            self.image_files = sorted([os.path.join(base_dir, x, y, z)
                                       for x in os.listdir(base_dir) if x.endswith('TrainDataset')
                                       for y in os.listdir(os.path.join(base_dir, x)) if y.endswith('images')
                                       for z in os.listdir(os.path.join(base_dir, x, y))])
            self.label_files = sorted([os.path.join(base_dir, x, y, z)
                                       for x in os.listdir(base_dir) if x.endswith('TrainDataset')
                                       for y in os.listdir(os.path.join(base_dir, x)) if y.endswith('masks')
                                       for z in os.listdir(os.path.join(base_dir, x, y))])
            self.image_files = sorted(list(set(self.image_files) - set(self.image_files[::5])))
            self.label_files = sorted(list(set(self.label_files) - set(self.label_files[::5])))
            self.sample_list = self.image_files
            print("Train total {} samples".format(len(self.sample_list)))

        elif self.split == "val":
            self.image_files = sorted([os.path.join(base_dir, x, y, z)
                                       for x in os.listdir(base_dir) if x.endswith('TrainDataset')
                                       for y in os.listdir(os.path.join(base_dir, x)) if y.endswith('images')
                                       for z in os.listdir(os.path.join(base_dir, x, y))])
            self.label_files = sorted([os.path.join(base_dir, x, y, z)
                                       for x in os.listdir(base_dir) if x.endswith('TrainDataset')
                                       for y in os.listdir(os.path.join(base_dir, x)) if y.endswith('masks')
                                       for z in os.listdir(os.path.join(base_dir, x, y))])
            self.image_files = sorted(self.image_files[::5])
            self.label_files = sorted(self.label_files[::5])
            self.sample_list = self.image_files
            print("val total {} samples".format(len(self.sample_list)))
        elif self.split == "test":
            self.image_files = sorted([os.path.join(base_dir, x, y, z)
                                       for x in os.listdir(base_dir) if x.endswith('TestDataset')
                                       for y in os.listdir(os.path.join(base_dir, x)) if y.endswith('images')
                                       for z in os.listdir(os.path.join(base_dir, x, y))])
            self.label_files = sorted([os.path.join(base_dir, x, y, z)
                                       for x in os.listdir(base_dir) if x.endswith('TestDataset')
                                       for y in os.listdir(os.path.join(base_dir, x)) if y.endswith('masks')
                                       for z in os.listdir(os.path.join(base_dir, x, y))])
            assert len(self.image_files) == len(self.label_files)
            self.sample_list = self.image_files
            print("Val total {} samples".format(len(self.sample_list)))
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            assert self.image_files[idx].split(self.sep)[-1][:-4] == self.label_files[idx].split(self.sep)[-1][:-4]
            img, mask = self.image_files[idx], self.label_files[idx]  # [224,224] [224,224]
            img = Image.open(img)
            mask = Image.open(mask)
            # img = img.convert('L')
            # mask = mask.convert('L')
            img = img.resize((256, 256), Image.LANCZOS)
            mask = mask.resize((256, 256), Image.NEAREST)
            image = np.array(img)
            # mask = np.array(mask,dtype=bool)
            # mask = mask.astype('uint8')
            mask = np.array(mask).astype('uint8')
            label = np.where(mask > 128, 1, 0)
            # label = torch.from_numpy(np.array(label)).unsqueeze(0)
        elif self.split == "val" or self.split == "test":
            assert self.image_files[idx].split(self.sep)[-1][:-4] == self.label_files[idx].split(self.sep)[-1][:-4]
            # image = io.imread(self.image_files[idx]).astype('uint8')
            # # image = numpy.array(image)
            # # image = Image.fromarray(image).convert('RGB').resize((512, 512))
            # label = io.imread(self.label_files[idx]).astype('bool').astype('uint8')
            img = self.image_files[idx]
            mask = self.label_files[idx]
            img = Image.open(img)
            mask = Image.open(mask)
            # img = img.convert('L')
            # mask = mask.convert('L')
            img = img.resize((256, 256), Image.LANCZOS)
            mask = mask.resize((256, 256), Image.NEAREST)
            image = np.array(img)
            # mask = np.array(mask,dtype=bool)
            # mask = mask.astype('uint8')
            mask = np.array(mask).astype('uint8')
            label = np.where(mask > 128, 1, 0)

            # label = torch.from_numpy(np.array(label)).unsqueeze(0)
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
