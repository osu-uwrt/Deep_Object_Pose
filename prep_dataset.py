import os
import glob
from tqdm import tqdm
import numpy as np

OBJECT = "soup"
TEST_RATIO = 0.1
NEGATIVE_RATIO = 0
DATASET_DIR = "/mnt/Data/fat/"
TRAIN_DIR = DATASET_DIR + OBJECT + "_train/"
TEST_DIR = DATASET_DIR + OBJECT + "_test/"
INCLUDE_RIGHT_IMAGE = True

if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

for f in glob.glob(TRAIN_DIR+"*"):
    os.remove(f)
for f in glob.glob(TEST_DIR+"*"):
    os.remove(f)

object_file = glob.glob(DATASET_DIR+"data/**/_object_settings.json", recursive=True)[0]
camera_file = glob.glob(DATASET_DIR+"data/**/_camera_settings.json", recursive=True)[0]
os.symlink(object_file, TRAIN_DIR+"_object_settings.json")
os.symlink(camera_file, TRAIN_DIR+"_camera_settings.json")
os.symlink(object_file, TEST_DIR+"_object_settings.json")
os.symlink(camera_file, TEST_DIR+"_camera_settings.json")
    

positives = []
negatives = []

print("Sorting posives and negatives")
for file_path in tqdm(glob.glob(DATASET_DIR + "data/**/*.left.json", recursive=True)):
    with open(file_path) as f:
        if OBJECT in f.read():
            positives.append(file_path)
        else:
            negatives.append(file_path)

np.random.shuffle(negatives)
samples = positives + negatives[:len(positives) * NEGATIVE_RATIO]
np.random.shuffle(samples)

num_test_samples = int(TEST_RATIO*len(samples))
test_samples = samples[:num_test_samples]
train_samples = samples[num_test_samples:]

print("Saving train set")
for path in tqdm(train_samples):
    src_path_without_extension = os.path.splitext(path)[0]
    src_path_without_extension_right = os.path.splitext(path)[0].replace("left", "right")
    dst_name = src_path_without_extension.replace(DATASET_DIR, "").replace("/", "_")
    dst_name_right = src_path_without_extension_right.replace(DATASET_DIR, "").replace("/", "_")

    os.symlink(src_path_without_extension+".json", TRAIN_DIR+dst_name+".json")
    os.symlink(src_path_without_extension+".jpg", TRAIN_DIR+dst_name+".jpg")
    if INCLUDE_RIGHT_IMAGE:
        os.symlink(src_path_without_extension_right+".json", TRAIN_DIR+dst_name_right+".json")
        os.symlink(src_path_without_extension_right+".jpg", TRAIN_DIR+dst_name_right+".jpg")

print("Saving test set")
for path in tqdm(test_samples):
    src_path_without_extension = os.path.splitext(path)[0]
    src_path_without_extension_right = os.path.splitext(path)[0].replace("left", "right")
    dst_name = src_path_without_extension.replace(DATASET_DIR, "").replace("/", "_")
    dst_name_right = src_path_without_extension_right.replace(DATASET_DIR, "").replace("/", "_")

    os.symlink(src_path_without_extension+".json", TEST_DIR+dst_name+".json")
    os.symlink(src_path_without_extension+".jpg", TEST_DIR+dst_name+".jpg")
    if INCLUDE_RIGHT_IMAGE:
        os.symlink(src_path_without_extension_right+".json", TEST_DIR+dst_name_right+".json")
        os.symlink(src_path_without_extension_right+".jpg", TEST_DIR+dst_name_right+".jpg")

