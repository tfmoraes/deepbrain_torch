import argparse
import itertools
import pathlib
import sys
import time
import typing

import nibabel as nb
import numpy as np
import torch
import vtk
from tqdm import tqdm
from vtk.util import numpy_support

from constants import BATCH_SIZE, OVERLAP, SIZE
from model import Unet3D

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-w",
    "--weights",
    type=pathlib.Path,
    metavar="path",
    help="Weight path",
    dest="weights",
    required=True,
)
parser.add_argument(
    "-i",
    "--input",
    type=pathlib.Path,
    metavar="path",
    help="Nifti input file",
    dest="input_file",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    default="output.vti",
    type=pathlib.Path,
    metavar="path",
    help="VTI output file",
    dest="output_file",
)
parser.add_argument(
    "-d",
    "--device",
    default="",
    type=str,
    help="Which device to use: cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, xla, vulkan",
    dest="device",
)
parser.add_argument(
    "--ww",
    default=None,
    type=int,
    dest="window_width"
)
parser.add_argument(
    "--wl",
    default=None,
    type=int,
    dest="window_level"
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=BATCH_SIZE,
    type=int,
    dest="batch_size"
)


def image_normalize(
    image: np.ndarray,
    min_: float = 0.0,
    max_: float = 1.0,
    output_dtype: np.dtype = np.int16,
) -> np.ndarray:
    output = np.empty(shape=image.shape, dtype=output_dtype)
    imin, imax = image.min(), image.max()
    output[:] = (image - imin) * ((max_ - min_) / (imax - imin)) + min_
    return output


def get_LUT_value_255(image: np.ndarray, window: int, level: int) -> np.ndarray:
    shape = image.shape
    data_ = image.ravel()
    image = np.piecewise(
        data_,
        [
            data_ <= (level - 0.5 - (window - 1) / 2),
            data_ > (level - 0.5 + (window - 1) / 2),
        ],
        [0, 255, lambda data_: ((data_ - (level - 0.5)) / (window - 1) + 0.5) * (255)],
    )
    image.shape = shape
    return image


def gen_patches(
    image: np.ndarray, patch_size: int, overlap: int, batch_size: int = BATCH_SIZE
) -> typing.Iterator[typing.Tuple[float, np.ndarray, typing.Iterable]]:
    sz, sy, sx = image.shape
    i_cuts = list(
        itertools.product(
            range(0, sz - patch_size, patch_size - overlap),
            range(0, sy - patch_size, patch_size - overlap),
            range(0, sx - patch_size, patch_size - overlap),
        )
    )
    patches = []
    indexes = []
    for idx, (iz, iy, ix) in enumerate(i_cuts):
        ez = iz + patch_size
        ey = iy + patch_size
        ex = ix + patch_size
        patch = image[iz:ez, iy:ey, ix:ex]
        patches.append(patch)
        indexes.append(((iz, ez), (iy, ey), (ix, ex)))
        if len(patches) == batch_size:
            yield (idx + 1.0) / len(i_cuts), np.asarray(patches), indexes
            patches = []
            indexes = []
    if patches:
        yield 1.0, np.asarray(patches), indexes


def pad_image(image: np.ndarray, patch_size: int = SIZE) -> np.ndarray:
    sz, sy, sx = image.shape
    pad_z = int(np.ceil(sz / patch_size) * patch_size) - sz + OVERLAP
    pad_y = int(np.ceil(sy / patch_size) * patch_size) - sy + OVERLAP
    pad_x = int(np.ceil(sx / patch_size) * patch_size) - sx + OVERLAP
    padded_image = np.pad(image, ((0, pad_z), (0, pad_y), (0, pad_x)))
    return padded_image


def brain_segment(
    image: np.ndarray,
    model: torch.nn.Module,
    dev: torch.device,
    mean: float,
    std: float,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    dz, dy, dx = image.shape
    image = image_normalize(image, 0.0, 1.0, output_dtype=np.float32)
    padded_image = pad_image(image, SIZE)
    padded_image = (padded_image - mean) / std
    probability_array = np.zeros_like(padded_image, dtype=np.float32)
    sums = np.zeros_like(padded_image)
    pbar = tqdm()
    # segmenting by patches
    for completion, patches, indexes in gen_patches(
        padded_image, SIZE, OVERLAP, batch_size
    ):
        with torch.no_grad():
            pred = (
                model(
                    torch.from_numpy(patches.reshape(-1, 1, SIZE, SIZE, SIZE)).to(dev)
                )
                .cpu()
                .numpy()
            )
        for i, ((iz, ez), (iy, ey), (ix, ex)) in enumerate(indexes):
            probability_array[iz:ez, iy:ey, ix:ex] += pred[i, 0]
            sums[iz:ez, iy:ey, ix:ex] += 1
        pbar.set_postfix(completion=completion * 100)
        pbar.update()
    pbar.close()
    probability_array[:dz, :dy, :dx] /= sums[:dz, :dy, :dx]
    return np.array(probability_array[:dz, :dy, :dx])


def to_vtk(
    n_array: np.ndarray,
    spacing: typing.Tuple[float, float, float] = (1.0, 1.0, 1.0),
    slice_number: int = 0,
    orientation: str = "AXIAL",
    origin: typing.Tuple[float, float, float] = (0, 0, 0),
    padding: typing.Tuple[float, float, float] = (0, 0, 0),
) -> vtk.vtkImageData:
    if orientation == "SAGITTAL":
        orientation = "SAGITAL"

    try:
        dz, dy, dx = n_array.shape
    except ValueError:
        dy, dx = n_array.shape
        dz = 1

    px, py, pz = padding

    v_image = numpy_support.numpy_to_vtk(n_array.flat)

    if orientation == "AXIAL":
        extent = (
            0 - px,
            dx - 1 - px,
            0 - py,
            dy - 1 - py,
            slice_number - pz,
            slice_number + dz - 1 - pz,
        )
    elif orientation == "SAGITAL":
        dx, dy, dz = dz, dx, dy
        extent = (
            slice_number - px,
            slice_number + dx - 1 - px,
            0 - py,
            dy - 1 - py,
            0 - pz,
            dz - 1 - pz,
        )
    elif orientation == "CORONAL":
        dx, dy, dz = dx, dz, dy
        extent = (
            0 - px,
            dx - 1 - px,
            slice_number - py,
            slice_number + dy - 1 - py,
            0 - pz,
            dz - 1 - pz,
        )

    # Generating the vtkImageData
    image = vtk.vtkImageData()
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDimensions(dx, dy, dz)
    # SetNumberOfScalarComponents and SetScalrType were replaced by
    # AllocateScalars
    #  image.SetNumberOfScalarComponents(1)
    #  image.SetScalarType(numpy_support.get_vtk_array_type(n_array.dtype))
    image.AllocateScalars(numpy_support.get_vtk_array_type(n_array.dtype), 1)
    image.SetExtent(extent)
    image.GetPointData().SetScalars(v_image)

    image_copy = vtk.vtkImageData()
    image_copy.DeepCopy(image)

    return image_copy


def image_save(image: np.ndarray, filename: str):
    v_image = to_vtk(image)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(v_image)
    writer.SetFileName(filename)
    writer.Write()


def main():
    args, _ = parser.parse_known_args()
    input_file = args.input_file
    weights_file = args.weights
    output_file = args.output_file
    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nii_data = nb.load(str(input_file))
    image = nii_data.get_fdata()
    mean = 0.0
    std = 1.0
    model = Unet3D()
    checkpoint = torch.load(weights_file)
    try:
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            dmodel = torch.nn.DataParallel(model)
            dmodel.load_state_dict(checkpoint["model_state_dict"])
            model = dmodel.module
        mean = checkpoint["mean"]
        std = checkpoint["std"]
    except TypeError:
        model = checkpoint
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
    print(f"mean={mean}, std={std}, {image.min()=}, {image.max()=}, {args.window_width=}, {args.window_level=}")
    model = model.to(dev)
    model.eval()

    if args.window_width is not None and args.window_level is not None:
        image = get_LUT_value_255(image, args.window_width, args.window_level)
        print("ww wl", image.min(), image.max())

    #probability_array = brain_segment(image, model, dev, 0.0, 1.0)
    probability_array = brain_segment(image, model, dev, mean, std, args.batch_size)
    image_save(probability_array, str(output_file))


if __name__ == "__main__":
    main()
