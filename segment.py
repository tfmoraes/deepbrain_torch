import itertools
import sys
import time

import nibabel as nb
import numpy as np
import vtk
import torch
from vtk.util import numpy_support

SIZE = 48
OVERLAP = SIZE // 2 + 1


def image_normalize(image, min_=0.0, max_=1.0, output_dtype=np.int16):
    output = np.empty(shape=image.shape, dtype=output_dtype)
    imin, imax = image.min(), image.max()
    output[:] = (image - imin) * ((max_ - min_) / (imax - imin)) + min_
    return output


def get_LUT_value(data, window, level):
    shape = data.shape
    data_ = data.ravel()
    data = np.piecewise(
        data_,
        [
            data_ <= (level - 0.5 - (window - 1) / 2),
            data_ > (level - 0.5 + (window - 1) / 2),
        ],
        [
            0,
            window,
            lambda data_: ((data_ - (level - 0.5)) / (window - 1) + 0.5) * (window),
        ],
    )
    data.shape = shape
    return data


def gen_patches(image, patch_size, overlap):
    sz, sy, sx = image.shape
    i_cuts = list(
        itertools.product(
            range(0, sz, patch_size - OVERLAP),
            range(0, sy, patch_size - OVERLAP),
            range(0, sx, patch_size - OVERLAP),
        )
    )
    sub_image = np.empty(shape=(patch_size, patch_size, patch_size), dtype="float32")
    for idx, (iz, iy, ix) in enumerate(i_cuts):
        sub_image[:] = 0
        _sub_image = image[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]
        sz, sy, sx = _sub_image.shape
        sub_image[0:sz, 0:sy, 0:sx] = _sub_image
        ez = iz + sz
        ey = iy + sy
        ex = ix + sx

        yield (idx + 1.0) / len(i_cuts), sub_image, ((iz, ez), (iy, ey), (ix, ex))


def predict_patch(sub_image, patch, nn_model, dev, patch_size=SIZE):
    (iz, ez), (iy, ey), (ix, ex) = patch
    with torch.no_grad():
        sub_mask = nn_model(
            torch.from_numpy(sub_image.reshape(1, 1, patch_size, patch_size, patch_size)).to(dev)
        )
    return sub_mask.cpu().numpy().reshape(patch_size, patch_size, patch_size)[
        0 : ez - iz, 0 : ey - iy, 0 : ex - ix
    ]


def brain_segment(image):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load("weights.pth").to(dev)
    probability_array = np.zeros_like(image, dtype=np.float32)
    image = image_normalize(image, 0.0, 1.0, output_dtype=np.float32)
    sums = np.zeros_like(image)
    # segmenting by patches
    for completion, sub_image, patch in gen_patches(image, SIZE, OVERLAP):
        (iz, ez), (iy, ey), (ix, ex) = patch
        sub_mask = predict_patch(sub_image, patch, model, dev, SIZE)
        probability_array[iz:ez, iy:ey, ix:ex] += sub_mask
        sums[iz:ez, iy:ey, ix:ex] += 1
        print(completion)

    probability_array /= sums
    return probability_array


def to_vtk(
    n_array,
    spacing=(1.0, 1.0, 1.0),
    slice_number=0,
    orientation="AXIAL",
    origin=(0, 0, 0),
    padding=(0, 0, 0),
):
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
    t0 = time.time()
    probability_array = brain_segment(image)
    t1 = time.time()
    print(f"\n\nTime: {t1 - t0} seconds\n\n")
    image_save(image, "input.vti")
    image_save(probability_array, "output.vti")



if __name__ == "__main__":
    main()
