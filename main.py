import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from skimage import io, transform
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torchvision.utils import make_grid
from tqdm import tqdm

import datasets
import model
import multilabel
import paths
import utils

dataset = datasets.dataset
test_dataset = datasets.test_dataset

np.random.seed(123)

purge_ppl = utils.purge_ppl
capsamples = utils.capsamples
Kfolder = utils.Kfolder


# Datasets for training and validation
validation_split = 0.2
shuffle_dataset = True

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.shuffle(indices)



# Define the validation split and transform
valsplit = 0.2
lengths = [int(len(dataset) * (1 - valsplit)), int(len(dataset) * valsplit)]
training_dataset, validation_dataset = torch.utils.data.random_split(dataset, lengths)
print("Training samples:", len(training_dataset.indices))
print("Validation samples:", len(validation_dataset.indices))
training_dataset.transform = transform
validation_dataset.transform = transform


# Loaders for training ad validation
train_loader = torch.utils.data.DataLoader(
    dataset=training_dataset, batch_size=8, shuffle=True, num_workers=0
)
validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset, batch_size=8, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0
)


dataiter = iter(train_loader)
tensor, labels = dataiter.next()


def imshow(img):
    img = img / 2 + 0.5  # "undo normalize"
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


imshow(torchvision.utils.make_grid(tensor))


# Loss function, optimizer, and thresholding
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "gpu":
    model.cuda()

model = model.Net()
losses = [[], []]
testlosses = [[], []]


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def thresholding(df):
    """Adapt prediction to label counts"""
    label_tot = df.dataframe.sum(axis=0)
    labels_tot = label_tot.sum()
    coeff = label_tot / labels_tot
    invert = coeff.apply(lambda x: 1 - x)
    # func= np.exp(val)/sum(np.exp(val))
    return invert.values


label_weights = torch.FloatTensor(thresholding(dataset))
if device == "gpu":
    label_weights = label_weights.cuda()


# Initialize variables and loop
epochmax = 10
t = 0
tt = 0
for epoch in range(epochmax):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # input
        inputs, labels = data
        if device == "cpu":
            inputs = inputs.float()
        if device == "gpu":
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    losses[1] += [running_loss / 30]
    losses[0] += [t]
    print("Epoch %d : loss %.3f" % (epoch + 1, running_loss / 30))
    t += 1
    running_loss = 0.0

    # Validate after epoch
    running_loss_test = 0.0
    for i, data in enumerate(validation_loader, 0):
        # input
        inputs, labels = data
        inputs = inputs.float()
        if device == "gpu":
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        lossb = criterion(outputs, labels)

        lossb = loss * label_weights
        lossb = lossb.mean()

        running_loss_test += lossb.item()

    testlosses[1] += [running_loss_test / 30]
    testlosses[0] += [tt]
    print("Epoch %d : Test loss %.3f" % (epoch + 1, running_loss_test / 30))
    tt += 1
    running_loss = 0.0


print("Finished training")


# Plot loss
plt.subplots()
plt.plot(losses[0], losses[1], label="train", marker=".")
plt.plot(testlosses[0], testlosses[1], label="test", marker=".")
plt.grid(linestyle="-", linewidth=1)
plt.xticks(losses[0])
plt.xlabel("Timesplit")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Test network
ok_labels = list(0.0 for i in range(14))
real_labels = list(0.0 for i in range(14))


def coincidences(array1, array2):
    correct_labels = list(0.0 for i in range(14))
    total_labels = list(0.0 for i in range(14))
    # Threshold tensors
    t = nn.Threshold(0.40, 0)
    a1 = t(array1)
    for i in range(len(a1)):

        for j in range(len(a1[i])):
            if (a1[i][j] > 0) and (array2[i][j]) > 0:
                correct_labels[j] += 1
            elif (a1[i][j] < 0.3) and (array2[i][j]) < 0.3:
                correct_labels[j] += 1

            total_labels[j] += 1

    return [correct_labels, total_labels]


with torch.no_grad():
    for i, data in enumerate(validation_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        hits, total = coincidences(outputs, labels)
        real_labels += np.array(real_labels) + np.array(total)
        ok_labels += np.array(ok_labels) + np.array(hits)

for i in range(len(ok_labels)):
    print(
        "Precision of label: %s is %.5f"
        % (dataset.labels[i], 100 * ok_labels[i] / real_labels[i])
    )


# Plot classification
class_total = list(0.0 for i in range(14))
labels_estimated = np.array(class_total)
labels_of_set = list(0.0 for i in range(14))


def assign_labels(array1):
    class_total_f = list(0.0 for i in range(14))
    # threshold tensors
    t = nn.Threshold(0.4, 0)

    a1 = t(array1)
    for i in range(len(a1)):
        for j in range(len(a1[i])):
            if a1[i][j] > 0:
                class_total_f[j] += 1

    return class_total_f


with torch.no_grad():
    for i, data in enumerate(validation_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        for ls in labels:
            labels_to_add = []
            for ils in ls:
                labels_to_add += [ils.item()]
            labels_of_set = np.array(labels_of_set) + np.array(labels_to_add)

        outputs = model(images)
        a = assign_labels(outputs)
        a = np.array(a)
        labels_estimated = np.array(labels_estimated) + a


labels_estimated = labels_estimated
clean_labels = dataset.labels
clean_labels = list(map(lambda x: x.replace("Label_", ""), clean_labels))

ld = pd.DataFrame(
    {"Ground Truth": labels_of_set, "Classified": labels_estimated}, index=clean_labels
)
ax = ld.plot.bar(rot=0)
plt.legend()


# Show images
dataiter = iter(validation_loader)
images, labels = dataiter.next()

# Sample
imshow(torchvision.utils.make_grid(images[0]))

index = None
index = []
for i in range(len(labels[0])):
    if labels[0][i] > 0.25:
        index = index + [i]

print(
    "GroundTruth: ",
    " ".join("%5s" % dataset.labels[index[j]] for j in range(len(index))),
)

output = model(images.to(device))
outputt = nn.Threshold(0.5, 0)(torch.sigmoid(output))


index = []
for i in range(len(output[0])):
    if outputt[0][i] > 0.2:
        index = index + [i]


print(
    "Predicted: ", " ".join("%5s" % dataset.labels[index[j]] for j in range(len(index)))
)
