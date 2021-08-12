import itertools
import sys
import time
import typing

import nibabel as nb
import numpy as np
import torch
import vtk
from vtk.util import numpy_support

from constants import BATCH_SIZE, OVERLAP, SIZE
from model import Unet3D


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
    print(f"{padded_image.shape=}, {image.shape=}")
    return padded_image


def brain_segment(
    image: np.ndarray, model: torch.nn.Module, dev: torch.device
) -> np.ndarray:
    dz, dy, dx = image.shape
    image = image_normalize(image, 0.0, 1.0, output_dtype=np.float32)
    padded_image = pad_image(image, SIZE)
    probability_array = np.zeros_like(padded_image, dtype=np.float32)
    sums = np.zeros_like(padded_image)
    # segmenting by patches
    for completion, patches, indexes in gen_patches(
        padded_image, SIZE, OVERLAP, BATCH_SIZE
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
        print(completion)

    probability_array /= sums
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
    input_file = sys.argv[1]
    nii_data = nb.load(input_file)
    image = nii_data.get_fdata()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Unet3D()
    checkpoint = torch.load("weights/weight_026.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dev)
    t0 = time.time()
    probability_array = brain_segment(image, model, dev)
    t1 = time.time()
    print(f"\n\nTime: {t1 - t0} seconds\n\n")
    image_save(image, "input.vti")
    image_save(probability_array, "output.vti")


if __name__ == "__main__":
    main()
