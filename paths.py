
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

#
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
##
TRAINING_DIR = os.path.join(DATA_DIR, "training")
os.makedirs(TRAINING_DIR, exist_ok=True)
TEST_DIR = os.path.join(DATA_DIR, "test")
os.makedirs(TEST_DIR, exist_ok=True)
###

TRAINING_IMAGES_DIR = os.path.join(TRAINING_DIR, "images")
os.makedirs(TRAINING_IMAGES_DIR, exist_ok=True)
TRAINING_LABELS_DIR = os.path.join(TRAINING_DIR, "labels")
os.makedirs(TRAINING_LABELS_DIR, exist_ok=True)
###
TEST_IMAGES_DIR = os.path.join(TEST_DIR, "images")
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
TEST_LABELS_DIR = os.path.join(TEST_DIR, "labels")
os.makedirs(TEST_LABELS_DIR, exist_ok=True)