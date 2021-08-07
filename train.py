import argparse
import pathlib

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import nibabel as nb

from constants import BATCH_SIZE, EPOCHS, NUM_PATCHES, OVERLAP, SIZE
from model import Unet3D
from segment import brain_segment

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", action="store_true", help="use gpu", dest="use_gpu")
parser.add_argument("-c", "--continue", action="store_true", dest="continue_train")
parser.add_argument("-b", "--backend", help="Backend", dest="backend")
args, _ = parser.parse_known_args()


def calc_proportions(masks):
    sum_bg = 0.0
    sum_fg = 0.0
    for m in masks:
        sum_bg += (m < 0.5).sum()
        sum_fg += (m >= 0.5).sum()

    return 1.0 - (sum_bg / masks.size), 1.0 - (sum_fg / masks.size)


class HDF5Sequence:
    def __init__(self, filename, batch_size):
        self.f_array = h5py.File(filename, "r")
        self.x = self.f_array["images"]
        self.y = self.f_array["masks"]
        self.batch_size = batch_size

    def calc_proportions(self):
        sum_bg = self.f_array["bg"][()]
        sum_fg = self.f_array["fg"][()]
        return 1.0 - (sum_bg / self.y.size), 1.0 - (sum_fg / self.y.size)

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        idx_i = idx * self.batch_size
        idx_e = (idx + 1) * self.batch_size

        batch_x = self.x[idx_i:idx_e]
        batch_y = self.y[idx_i:idx_e]

        random_idx = np.arange(idx_e - idx_i)
        np.random.shuffle(random_idx)

        batch_x = np.array(batch_x[random_idx]).reshape(-1, 1, SIZE, SIZE, SIZE)
        batch_y = np.array(batch_y[random_idx]).reshape(-1, 1, SIZE, SIZE, SIZE)
        return batch_x, batch_y


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Unet3D()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(dev)
    model.train()

    image_test = nb.load("datasets/cc359/Original/CC0016_philips_15_55_M.nii.gz").get_fdata()

    training_files_gen = HDF5Sequence("train_arrays.h5", BATCH_SIZE)
    testing_files_gen = HDF5Sequence("test_arrays.h5", BATCH_SIZE)
    prop_bg, prop_fg = training_files_gen.calc_proportions()
    pos_weight = prop_fg / prop_bg
    print(f"proportion: {prop_fg}, {prop_bg}, {pos_weight}")

    # criterion = nn.BCELoss(weight=torch.from_numpy(np.array((0.1, 0.9))), reduction='mean')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(dev))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    writer = SummaryWriter()
    # writer.add_graph(model, torch.randn(1, 1, SIZE, SIZE, SIZE).to(dev))
    print(f"{len(training_files_gen)}, {training_files_gen.x.shape[0]}, {BATCH_SIZE}")
    # 1/0

    best_loss = 10000

    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        total_loss = 0
        total_correct = 0
        total_size = 0
        t = trange(len(training_files_gen))
        for i, (img, mask) in zip(t, training_files_gen):
            size = mask.size
            total_size += size

            img = torch.from_numpy(img).to(dev)
            mask = torch.from_numpy(mask).to(dev)
            mask_pred = model(img)
            loss = criterion(mask_pred, mask)

            total_loss += loss.item()
            correct = float((torch.sum((mask_pred - mask) ** 2) ** (0.5)).float())
            total_correct += correct

            t.set_postfix(loss=loss.item(), correct=correct / size)

            # print(
            #     f"{epoch:03}/{EPOCHS:03} - {i:03}/{len(training_files_gen):03}, {loss.item():03.5f}, {correct / size:03.5f}"
            # )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss)

        dz, dy, dx = image_test.shape
        output_test = brain_segment(image_test, model, dev)

        actual_loss = total_loss / len(training_files_gen)
        actual_acc = total_correct / total_size
        writer.add_scalar("Loss", actual_loss, epoch)
        writer.add_scalar("Correct", total_correct, epoch)
        writer.add_scalar("Accuracy", actual_acc, epoch)
        writer.add_image("View 1", output_test.max(0).reshape(1, dy, dx), epoch)
        writer.add_image("View 2", output_test.max(1).reshape(1, dz, dx), epoch)
        writer.add_image("View 3", output_test.max(2).reshape(1, dz, dy), epoch)

        if actual_loss <= best_loss:
            torch.save(model, "weights.pth")
            best_loss = actual_loss

    writer.flush()


def main():
    train()


if __name__ == "__main__":
    main()
