import itertools
import os
import pathlib
import random
import sys
import math
import multiprocessing
from tempfile import mktemp
import concurrent.futures

import h5py
import nibabel as nb
import numpy as np
from scipy.ndimage import rotate
from skimage.transform import resize

import file_utils
from constants import BATCH_SIZE, EPOCHS, NUM_PATCHES, OVERLAP, SIZE, STEP_ROT, NUM_ROTS
from utils import apply_transform, image_normalize

MIRRORING = 2


def get_image_patch(image, patch, patch_size):
    sub_image = np.zeros(shape=(patch_size, patch_size, patch_size), dtype="float32")
    iz, iy, ix = patch

    _sub_image = image[iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size]

    sz, sy, sx = _sub_image.shape
    sub_image[0:sz, 0:sy, 0:sx] = _sub_image
    return sub_image


def gen_image_patches(files, queue, patch_size=SIZE, num_patches=NUM_PATCHES):
    image_filename, mask_filename = files
    original_image = nb.load(str(image_filename)).get_fdata()
    original_mask = nb.load(str(mask_filename)).get_fdata()

    normalized_image = image_normalize(original_image)
    normalized_mask = image_normalize(original_mask)

    patches_files = []
    # Mirroring
    for m in range(MIRRORING):
        if m == 0:
            image = normalized_image.copy()
            mask = normalized_mask.copy()
        elif m == 1:
            image = normalized_image[:, :, ::-1].copy()
            mask = normalized_mask[:, :, ::-1].copy()
        elif m == 2:
            image = normalized_image[:, ::-1, :].copy()
            mask = normalized_mask[:, ::-1, :].copy()
        elif m == 3:
            image = normalized_image[::-1].copy()
            mask = normalized_mask[::-1].copy()

        for n in range(NUM_ROTS):
            if n == 0:
                _image = image
                _mask = mask
            else:
                rot1 = random.randint(1, 359)
                rot2 = random.randint(1, 359)
                rot3 = random.randint(1, 359)

                _image = apply_transform(image, rot1, rot2, rot3)
                _mask = apply_transform(mask, rot1, rot2, rot3)

            if _image is None or _mask is None:
                continue

            sz, sy, sx = _image.shape
            patches = list(
                itertools.product(
                    range(0, sz, patch_size - OVERLAP),
                    range(0, sy, patch_size - OVERLAP),
                    range(0, sx, patch_size - OVERLAP),
                )
            )
            random.shuffle(patches)

            patches_added = 0
            disponible_patches = []
            for patch in patches:
                sub_mask = get_image_patch(_mask, patch, patch_size)
                if (sub_mask.sum()) >= (sub_mask.size * 0.1):
                    sub_image = get_image_patch(_image, patch, patch_size)
                    queue.put((sub_image, sub_mask))
                    patches_added += 1
                else:
                    disponible_patches.append(patch)
                if patches_added >= math.ceil(num_patches * 0.80):
                    print("> 10%%", patches_added)
                    break

            patches = disponible_patches
            disponible_patches = []
            for patch in patches:
                sub_mask = get_image_patch(_mask, patch, patch_size)
                if (sub_mask.sum()) < (sub_mask.size * 0.1):
                    sub_image = get_image_patch(_image, patch, patch_size)
                    queue.put((sub_image, sub_mask))
                    patches_added += 1
                else:
                    disponible_patches.append(patch)
                if patches_added >= num_patches:
                    print("< 10%%", patches_added)
                    break

            if patches_added < num_patches:
                for patch in disponible_patches:
                    sub_mask = get_image_patch(_mask, patch, patch_size)
                    sub_image = get_image_patch(_image, patch, patch_size)
                    queue.put((sub_image, sub_mask))
                    patches_added += 1
                    if patches_added >= num_patches:
                        print("outros", patches_added)
                        break

    return patches_files


def gen_all_patches(files):
    m = multiprocessing.Manager()
    queue = m.Queue(maxsize=10000)
    total_size = len(files) * NUM_ROTS * MIRRORING * NUM_PATCHES
    train_size  = int(total_size * 0.8)
    test_size = total_size - train_size
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        f_h5 = executor.submit(h5file_from_patches, train_size, test_size, queue, SIZE)
        futures = []
        for image_file, mask_file in files:
            f = executor.submit(gen_image_patches, (image_file, mask_file), queue, SIZE, NUM_PATCHES)
            futures.append(f)
        # futures.append(f_h5)
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
        queue.put(None)
        print(f_h5.result())


def h5file_from_patches(train_size, test_size, queue, patch_size=SIZE):
    print(train_size, test_size, queue, patch_size)
    f_train = h5py.File("train_arrays.h5", "w")
    train_images = f_train.create_dataset(
        "images", (train_size, patch_size, patch_size, patch_size, 1), dtype="float32"
    )
    train_masks = f_train.create_dataset(
        "masks", (train_size, patch_size, patch_size, patch_size, 1), dtype="float32"
    )
    f_train["bg"] = 0
    f_train["fg"] = 0
    f_test = h5py.File("test_arrays.h5", "w")
    test_images = f_test.create_dataset(
        "images", (test_size, patch_size, patch_size, patch_size, 1), dtype="float32"
    )
    test_masks = f_test.create_dataset(
        "masks", (test_size, patch_size, patch_size, patch_size, 1), dtype="float32"
    )
    f_test["bg"] = 0
    f_test["fg"] = 0

    indexes = [["train", i] for i in range(train_size)] + [["test", i] for i in range(test_size)]
    random.shuffle(indexes)
    i = 0
    while True:
        print(f"{i}/{train_size + test_size}")
        value = queue.get()
        if value is None:
            break
        image = value[0].reshape(patch_size, patch_size, patch_size, 1)
        mask = value[1].reshape(patch_size, patch_size, patch_size, 1)
        arr, idx = indexes[i]
        i+=1
        if arr == "train":
            train_images[idx] = image
            train_masks[idx] = mask
            f_train["bg"][()] += (mask < 0.5).sum()
            f_train["fg"][()] += (mask >= 0.5).sum()
        else:
            test_images[idx] = image
            test_masks[idx] = mask
            f_test["bg"][()] += (mask < 0.5).sum()
            f_test["fg"][()] += (mask >= 0.5).sum()

    f_train["mean"] = np.mean(f_train["images"])
    f_train["std"] = np.mean(f_train["images"])

    f_test["mean"] = np.mean(f_test["images"])
    f_test["std"] = np.mean(f_test["images"])


def main():
    deepbrain_folder = pathlib.Path("datasets").resolve()
    # deepbrain_folder = pathlib.Path("datasets").resolve()
    # files = file_utils.get_lidc_filenames(deepbrain_folder)

    cc359_files = file_utils.get_cc359_filenames(deepbrain_folder)
    nfbs_files = file_utils.get_nfbs_filenames(deepbrain_folder)
    files = cc359_files + nfbs_files

    patches_files = gen_all_patches(files)
    # random.shuffle(patches_files)

    # training_files = patches_files[: int(len(patches_files) * 0.80)]
    # testing_files = patches_files[int(len(patches_files) * 0.80) :]

    # h5file_from_patches(training_files, "train_arrays.h5")
    # h5file_from_patches(testing_files, "test_arrays.h5")


if __name__ == "__main__":
    main()
