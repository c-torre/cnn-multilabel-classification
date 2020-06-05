"""
Make the multi-labels DataFrame
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import paths

tqdm.pandas()


def _make_dataset_index(images_dir):
    """ Make index as long as numbe of images """

    def image_id_to_name(element, prefix="im", suffix=".jpg"):
        """ Output full image name """
        return prefix + str(element) + suffix

    num_images_index = pd.Series(range(len(os.listdir(images_dir))), name="image_name")
    num_images_index += 1  # Images are sadly indexed from 1
    return num_images_index.progress_apply(image_id_to_name)


def load_and_make_multilabel(images_dir, labels_file_dir):

    # Load the csv with first column as index
    data_info = pd.read_csv(labels_file_dir, index_col=0)
    # Put labels as columns and add a "1" for matching image name
    dummies = pd.get_dummies(data_info, prefix="", prefix_sep="", dtype=int)
    # Add all labels for an image together
    dummies = dummies.groupby("image_name").sum()
    # Reindex to have all images in the directory covered, with labels as "0"
    dummies = dummies.reindex(_make_dataset_index(images_dir), fill_value=0)
    # Sanity checks for fake bool DataFrame
    assert dummies.max().max() == 1 and dummies.min().min() == 0

    return dummies.sort_index()


def smart_save_multilabels(df, path, force=False):
    if not os.path.exists(path) or force:
        df.to_csv(path)


if __name__ == "__main__":
    debug = True
    if debug:
        IMAGES_DIR = paths.TRAINING_IMAGES_DIR
        LABELS_DIR = paths.TRAINING_LABELS_DIR

        LABELS_SINGLE = os.path.join(LABELS_DIR, "labels_single.csv")
        LABELS_MULTI = os.path.join(LABELS_DIR, "labels_multi.csv")
    image_labels = load_and_make_multilabel(IMAGES_DIR, LABELS_SINGLE)
    smart_save_multilabels(image_labels, LABELS_MULTI)
