import os

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from skimage import io, transform
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torchvision.utils import make_grid

import paths


class CustomDataset(Dataset):
    def __init__(self, img_path, csv_path, transform=None):

        self.dataframe = pd.read_csv(csv_path, index_col=0)
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):

        image_name = f"im{index+1}.jpg"  # Bad indexing at 1
        image_path = os.path.join(self.img_path, image_name)
        img_values = Image.open(image_path).convert("RGB")

        if self.transform:
            img_values = self.transform(img_values)

        single_image_labels = np.array(self.dataframe.loc[image_name])

        sample = [img_values, single_image_labels]
        return sample

    def __len__(self):
        return len(self.dataframe.index)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(image.shape) < 3:
            image = np.stack((image,) * 3, axis=-1)
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "label": label}


class ToRGB(object):
    def __call__(self, sample):
        if len(sample.getbands()) < 3:
            sample = [sample] + [sample] + [sample]

        return sample


transform = transforms.Compose(
    [
        ToRGB(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)  # , ToRGB() ,transforms.Resize(64), transforms.RandomResizedCrop(128),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)

# Training dataset
TRAINING_LABELS_DIR = paths.TRAINING_LABELS_DIR
data_info = os.path.join(TRAINING_LABELS_DIR, "labels_multi.csv")
dataset = CustomDataset(paths.TRAINING_IMAGES_DIR, data_info, transform)

# Test dataset
empty_test = pd.DataFrame(
    data=0,
    index=[f"im{idx+1}.jpg" for idx in range(5000)],
    columns=(pd.read_csv(data_info, index_col=0)).columns,
)
save_path = os.path.join(paths.TEST_LABELS_DIR, "labels_multi.csv")
empty_test.to_csv(save_path)
#%%
test_dataset = CustomDataset(paths.TRAINING_IMAGES_DIR, save_path, transform)
