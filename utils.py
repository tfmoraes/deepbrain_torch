import numpy as np
import transformations
from scipy.ndimage import rotate
from skimage.transform import resize
import transforms

def apply_transform(image, r1, r2, r3):
    r1 = np.deg2rad(r1)
    r2 = np.deg2rad(r2)
    r3 = np.deg2rad(r3)
    cz, cy, cx = [i // 2 + 1 for i in image.shape]
    T0 = transformations.translation_matrix((-cz, -cy, -cx))
    Rz = transformations.rotation_matrix(r1, (1, 0, 0))
    Ry = transformations.rotation_matrix(r2, (0, 1, 0))
    Rx = transformations.rotation_matrix(r3, (0, 0, 1))
    T1 = transformations.translation_matrix((cz, cy, cx))
    M = transformations.concatenate_matrices(T1, Rz, Ry, Rx, T0)

    out = np.zeros_like(image)
    transforms.apply_view_matrix_transform(
        image, (1, 1, 1), M, 0, "AXIAL", 1, image.min(), out
    )
    return out




def image_normalize(image, min_=0.0, max_=1.0):
    imin, imax = image.min(), image.max()
    if imin == imax:
        return None
    return (image - imin) * ((max_ - min_) / (imax - imin)) + min_


def get_plaidml_devices(gpu=False, _id=0):
    import plaidml
    ctx = plaidml.Context()
    plaidml.settings._setup_for_test(plaidml.settings.user_settings)
    plaidml.settings.experimental = True
    devices, _ = plaidml.devices(ctx, limit=100, return_all=True)
    if gpu:
        for device in devices:
            if b'cuda' in device.description.lower():
                return device
        for device in devices:
            if b'opencl' in device.description.lower() and device.id.endswith(b'%d' % _id):
                return device
    for device in devices:
        if b'llvm' in device.description.lower():
            return device
