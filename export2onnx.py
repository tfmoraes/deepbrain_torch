import argparse
from pathlib import Path

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
    torch.onnx.export(
        torch_model,
        example_input,
        args.output,
        verbose=True,
        input_names=[
            "input",
        ],
        output_names=["output",],
    )


if __name__ == "__main__":
    main()
