import argparse
import pathlib
import shutil
import typing

import h5py
import nibabel as nb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from constants import SIZE
from loss import DiceBCELoss, DiceLoss, FocalLoss, TverskyLoss
from model import Unet3D
from segment import brain_segment

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-d",
    "--device",
    default="",
    type=str,
    help="Which device to use: cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, xla, vulkan",
    dest="device",
)
parser.add_argument(
    "-e",
    "--epochs",
    default=200,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-c",
    "--continue",
    help="Resume training",
    action="store_true",
    dest="continue_train",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=26,
    type=int,
    metavar="N",
    help="Batch size"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="Learning rate",
    dest="lr",
)
parser.add_argument(
    "--early-stop",
    default=0,
    type=int,
    metavar="N",
    help="Number of epochs of no improvement to early stop. If 0 then early-stop is not activated.",
    dest="early_stop",
)
parser.add_argument(
    "--patience",
    default=25,
    type=int,
    metavar="N",
    help="Number of epochs necessary before stop by early stop",
    dest="patience",
)
args, _ = parser.parse_known_args()


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
    mean: float,
    std: float,
    is_best: bool = False,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "mean": mean,
        "std": std,
    }
    f_path = pathlib.Path("checkpoints/checkpoint.pt").resolve()
    f_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(f_path))
    if is_best:
        f_path_best_weight = pathlib.Path(f"weights/weight_{epoch:03}.pt").resolve()
        f_path_best_weight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f_path, f_path_best_weight)


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer
) -> typing.Tuple[int, nn.Module, optim.Optimizer, float]:
    f_path = pathlib.Path("checkpoints/checkpoint.pt").resolve()
    checkpoint = torch.load(str(f_path))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    return epoch, model, optimizer, best_loss


def calc_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    with torch.no_grad():
        y_pred = y_pred >= 0.5
        y_true = y_true >= 0.5
        acc = y_pred.eq(y_true).sum().item() / y_true.numel()
        return acc


class HDF5Sequence:
    def __init__(self, filename: str, batch_size: int):
        self.f_array = h5py.File(filename, "r")
        self.x = self.f_array["images"]
        self.y = self.f_array["masks"]
        self.mean = float(self.f_array["mean"][()])
        self.std = float(self.f_array["std"][()])
        self.batch_size = batch_size

    def calc_proportions(self) -> typing.Tuple[float, float]:
        sum_bg = self.f_array["bg"][()]
        sum_fg = self.f_array["fg"][()]
        return 1.0 - (sum_bg / self.y.size), 1.0 - (sum_fg / self.y.size)

    def __len__(self) -> int:
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx) -> typing.Tuple[np.ndarray, np.ndarray]:
        idx_i = idx * self.batch_size
        idx_e = (idx + 1) * self.batch_size

        batch_x = self.x[idx_i:idx_e]
        batch_y = self.y[idx_i:idx_e]

        random_idx = np.arange(idx_e - idx_i)
        np.random.shuffle(random_idx)

        batch_x = np.array(batch_x[random_idx]).reshape(-1, 1, SIZE, SIZE, SIZE)
        batch_y = np.array(batch_y[random_idx]).reshape(-1, 1, SIZE, SIZE, SIZE)
        return batch_x, batch_y


def train():
    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Unet3D()
    writer = SummaryWriter()
    writer.add_graph(model, torch.randn(1, 1, SIZE, SIZE, SIZE))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(dev)

    image_test = nb.load(
        "datasets/cc359/Original/CC0016_philips_15_55_M.nii.gz"
    ).get_fdata()

    training_files_gen = HDF5Sequence("train_arrays.h5", args.batch_size)
    testing_files_gen = HDF5Sequence("test_arrays.h5", args.batch_size)
    # mean = training_files_gen.mean
    # std = training_files_gen.std
    mean = 0.0
    std = 1.0
    prop_bg, prop_fg = training_files_gen.calc_proportions()
    pos_weight = prop_fg / prop_bg

    print(f"proportion: {prop_fg}, {prop_bg}, {pos_weight}")

    print(
        f"{len(training_files_gen)}, {training_files_gen.x.shape[0]}, {args.batch_size}"
    )

    criterion = DiceLoss(apply_sigmoid=False)
    # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(dev))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.continue_train:
        epoch, model, optimizer, best_loss = load_checkpoint(model, optimizer)
        start_epoch = epoch + 1
    else:
        start_epoch = 0
        best_loss = 100000

    epochs_no_improve = 0
    for epoch in trange(start_epoch, args.epochs):
        losses = {
            "train": [],
            "validate": [],
        }
        accuracies = {
            "train": [],
            "validate": [],
        }
        for stage in ("train", "validate"):
            if stage == "train":
                model.train()
                t = trange(len(training_files_gen))
            else:
                model.eval()
                t = trange(len(testing_files_gen))

            for i, (x, y_true) in zip(t, training_files_gen):
                x = (x - mean) / std
                x = torch.from_numpy(x).to(dev)
                y_true = torch.from_numpy(y_true).to(dev)
                optimizer.zero_grad()

                with torch.set_grad_enabled(stage == "train"):
                    y_pred = model(x)
                    loss = criterion(y_pred, y_true)
                    accuracy = 100 * calc_accuracy(y_pred, y_true)

                    losses[stage].append(loss.item())
                    accuracies[stage].append(accuracy)

                    t.set_postfix(loss=loss.item(), accuracy=accuracy, stage=stage)

                    if stage == "train":
                        loss.backward()
                        optimizer.step()

        dz, dy, dx = image_test.shape
        output_test = brain_segment(image_test, model, dev, mean, std)
        print("Min max", output_test.min(), output_test.max())
        output_test = (output_test > 0.75) * 1.0

        actual_loss = np.mean(losses["train"])

        writer.add_scalar("Loss train", actual_loss, epoch)
        writer.add_scalar("Accuracy train", np.mean(accuracies["train"]), epoch)

        writer.add_scalar("Loss validation", np.mean(losses["validate"]), epoch)
        writer.add_scalar("Accuracy validation", np.mean(accuracies["validate"]), epoch)

        writer.add_image("View 1", output_test.max(0).reshape(1, dy, dx), epoch)
        writer.add_image("View 2", output_test.max(1).reshape(1, dz, dx), epoch)
        writer.add_image("View 3", output_test.max(2).reshape(1, dz, dy), epoch)

        if actual_loss <= best_loss:
            best_loss = actual_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint(
            epoch,
            model,
            optimizer,
            best_loss,
            mean,
            std,
            is_best=actual_loss == best_loss,
        )

        if epoch >= args.patience and args.early_stop > 0 and epochs_no_improve == args.early_stop:
            print("Early-top!")
            break

    writer.flush()


def main():
    train()


if __name__ == "__main__":
    main()
