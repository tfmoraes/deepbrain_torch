import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import SIZE


class DeepBrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_network()

    def build_network(self):
        self.n_n_conv3d_1 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=8,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=1,
            bias=True,
        )
        self.n_n_conv3d_2 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=8,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=8,
            bias=True,
        )
        self.n_n_conv3d_2_Relu_0_pooling = nn.MaxPool3d(
            dilation=1,
            kernel_size=[2, 2, 2],
            ceil_mode=False,
            stride=[2, 2, 2],
            return_indices=False,
        )
        self.n_n_conv3d_3 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=16,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=8,
            bias=True,
        )
        self.n_n_conv3d_4 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=16,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=16,
            bias=True,
        )
        self.n_n_conv3d_4_Relu_0_pooling = nn.MaxPool3d(
            dilation=1,
            kernel_size=[2, 2, 2],
            ceil_mode=False,
            stride=[2, 2, 2],
            return_indices=False,
        )
        self.n_n_conv3d_5 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=32,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=16,
            bias=True,
        )
        self.n_n_conv3d_6 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=32,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=32,
            bias=True,
        )
        self.n_n_conv3d_6_Relu_0_pooling = nn.MaxPool3d(
            dilation=1,
            kernel_size=[2, 2, 2],
            ceil_mode=False,
            stride=[2, 2, 2],
            return_indices=False,
        )
        self.n_n_conv3d_transpose_1 = nn.ConvTranspose3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=32,
            padding=[1, 1, 1],
            output_padding=[0, 0, 0],
            kernel_size=(4, 4, 4),
            stride=[2, 2, 2],
            in_channels=32,
            bias=False,
        )
        self.n_n_conv3d_7 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=32,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=64,
            bias=True,
        )
        self.n_n_conv3d_transpose_2 = nn.ConvTranspose3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=16,
            padding=[1, 1, 1],
            output_padding=[0, 0, 0],
            kernel_size=(4, 4, 4),
            stride=[2, 2, 2],
            in_channels=32,
            bias=False,
        )
        self.n_n_conv3d_8 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=16,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=32,
            bias=True,
        )
        self.n_n_conv3d_transpose_3 = nn.ConvTranspose3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=8,
            padding=[1, 1, 1],
            output_padding=[0, 0, 0],
            kernel_size=(4, 4, 4),
            stride=[2, 2, 2],
            in_channels=16,
            bias=False,
        )
        self.n_n_conv3d_9 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=8,
            padding=[2, 2, 2],
            kernel_size=(5, 5, 5),
            stride=[1, 1, 1],
            in_channels=16,
            bias=True,
        )
        self.n_n_conv3d_10 = nn.Conv3d(
            groups=1,
            dilation=[1, 1, 1],
            out_channels=1,
            padding=[0, 0, 0],
            kernel_size=(1, 1, 1),
            stride=[1, 1, 1],
            in_channels=8,
            bias=True,
        )

        self.n_n_linear = nn.Linear(SIZE, SIZE)

    def forward(self, img):
        t_adjusted_input4 = img.permute(*[0, 4, 1, 2, 3])
        t_convolution_output4 = self.n_n_conv3d_1(t_adjusted_input4)
        t_conv3d_1_Relu_0 = F.relu(t_convolution_output4)
        t_convolution_output3 = self.n_n_conv3d_2(t_conv3d_1_Relu_0)
        t_conv3d_2_Relu_0 = F.relu(t_convolution_output3)
        t_conv3d_2_Relu_0_pooling0 = self.n_n_conv3d_2_Relu_0_pooling(t_conv3d_2_Relu_0)
        t_convolution_output8 = self.n_n_conv3d_3(t_conv3d_2_Relu_0_pooling0)
        t_conv3d_3_Relu_0 = F.relu(t_convolution_output8)
        t_convolution_output7 = self.n_n_conv3d_4(t_conv3d_3_Relu_0)
        t_conv3d_4_Relu_0 = F.relu(t_convolution_output7)
        t_conv3d_4_Relu_0_pooling0 = self.n_n_conv3d_4_Relu_0_pooling(t_conv3d_4_Relu_0)
        t_convolution_output12 = self.n_n_conv3d_5(t_conv3d_4_Relu_0_pooling0)
        t_conv3d_5_Relu_0 = F.relu(t_convolution_output12)
        t_convolution_output11 = self.n_n_conv3d_6(t_conv3d_5_Relu_0)
        t_conv3d_6_Relu_0 = F.relu(t_convolution_output11)
        t_conv3d_6_Relu_0_pooling0 = self.n_n_conv3d_6_Relu_0_pooling(t_conv3d_6_Relu_0)
        t_convolution_output10 = self.n_n_conv3d_transpose_1(t_conv3d_6_Relu_0_pooling0)
        t_concatenate_1_concat_0 = torch.cat(
            (t_convolution_output10, t_conv3d_6_Relu_0), dim=1
        )
        t_convolution_output9 = self.n_n_conv3d_7(t_concatenate_1_concat_0)
        t_conv3d_7_Relu_0 = F.relu(t_convolution_output9)
        t_convolution_output6 = self.n_n_conv3d_transpose_2(t_conv3d_7_Relu_0)
        t_concatenate_2_concat_0 = torch.cat(
            (t_convolution_output6, t_conv3d_4_Relu_0), dim=1
        )
        t_convolution_output5 = self.n_n_conv3d_8(t_concatenate_2_concat_0)
        t_conv3d_8_Relu_0 = F.relu(t_convolution_output5)
        t_convolution_output2 = self.n_n_conv3d_transpose_3(t_conv3d_8_Relu_0)
        t_concatenate_3_concat_0 = torch.cat(
            (t_convolution_output2, t_conv3d_2_Relu_0), dim=1
        )
        t_convolution_output1 = self.n_n_conv3d_9(t_concatenate_3_concat_0)
        t_conv3d_9_Relu_0 = F.relu(t_convolution_output1)
        t_convolution_output = self.n_n_conv3d_10(t_conv3d_9_Relu_0)
        t_push_transpose_out_0 = t_convolution_output.permute(*[0, 2, 3, 4, 1])
        t_linear_out = self.n_n_linear(t_push_transpose_out_0.view(-1, SIZE)).view(t_push_transpose_out_0.shape)
        t_dense_1 = torch.sigmoid(t_linear_out)
        return t_dense_1


def main():
    dev = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
    model = DeepBrainModel()
    model.to(dev)
    model.eval()
    print(next(model.parameters()).is_cuda)  # True
    img = torch.randn(1, SIZE, SIZE, SIZE, 1).to(dev)
    with torch.no_grad():
        out = model(img)
    print(out.shape)


if __name__ == "__main__":
    main()
