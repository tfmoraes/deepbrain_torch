import argparse
from pathlib import Path

import coremltools as ct
import torch

from constants import SIZE
from model import Unet3D

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input",
    type=Path,
    help="Torch weights file",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    type=Path,
    help="Coreml output file",
    required=True,
)

args, _ = parser.parse_known_args()
print(args)


def rename_output(ct_model: ct.models.MLModel) -> ct.models.MLModel:
    spec = ct_model.get_spec()
    output_names = [out.name for out in spec.description.output]
    ct.utils.rename_feature(spec, output_names[0], "output")
    ct_model = ct.models.MLModel(spec)
    return ct_model


def insert_metadata(ct_model: ct.models.MLModel):
    ct_model.input_description["input"] = "Input image to be segmented"
    ct_model.output_description["output"] = "Image segmented"
    ct_model.author = "Thiago Franco de Moraes"
    ct_model.license = (
        "Please see https://github.com/tfmoraes/deepbrain_torch for license information"
    )
    ct_model.short_description = "Unet 3D to segment brain in MRI T1 images"
    ct_model.version = "1.1.0"


def main():
    state_dict = torch.load(str(args.input), map_location=torch.device("cpu"))
    torch_model = Unet3D()
    try:
        torch_model.load_state_dict(state_dict["model_state_dict"])
    except RuntimeError as err:
        parallel_model = torch.nn.DataParallel(torch_model)
        parallel_model.load_state_dict(state_dict["model_state_dict"])
        torch_model.load_state_dict(parallel_model.module.state_dict())
    torch_model.eval()

    example_input = torch.randn(1, 1, SIZE, SIZE, SIZE)
    traced_model = torch.jit.trace(torch_model, example_input)
    out = traced_model(example_input)

    ct_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    )

    ct_model = rename_output(ct_model)
    insert_metadata(ct_model)

    ct_model.save(str(args.output))


if __name__ == "__main__":
    main()
