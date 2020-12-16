
import torch
from torch import nn


class DilatedCausalConv(nn.Module):
    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv, self).__init__()

        self.conv = nn.Conv1d(channels, channels,
                            kernel_size = 2, stride=1,
                            dilation=dilation,
                            padding=0,
                            bias=False)

    def forward(self, x):
        output = self.conv(x)
        return output


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CausalConv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                            kernel_size=2, stride=1, padding=1,
                            bias=False)


    def forward(self, x):
        output = self.conv(x)

        return output[:, :, :-1]


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation):
        super(ResidualBlock, self).__init__()

        self.dilation = dilation
        self.dilated = DilatedCausalConv(residual_channels, dilation=dilation)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

        self.filter_tanh = nn.Tanh()
        self.gate_sig = nn.Sigmoid()


    def forward(self, x, skip_size):
        output = self.dilated(x)

        filter = self.filter_tanh(output)
        gate = self.gate_sig(output)
        activated = gate * filter

        output = self.residual_conv(activated)
        cut_input = x[:, :, -output.shape[2]:]
        output += cut_input

        skip = self.skip_conv(activated)
        skip = skip[:, :, -skip_size:]

        return output, skip

class BlockStack(nn.Module):
    def __init__(self, layer_size, stack_size, residual_channels, skip_channels):
        super(BlockStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.blocks = self.stack_blocks(residual_channels, skip_channels)

    def _dilations(self):
        dilations = []

        for stack in range(0, self.stack_size):
            for layer in range(0, self.layer_size):
                dilations.append(2 ** layer)

        return dilations


    def stack_blocks(self, residual_channels, skip_channels):
        return [ResidualBlock(residual_channels, skip_channels, dilation) for dilation in self._dilations()]


    def forward(self, x, skip_size):
        output = x
        skips = []

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            output, skip = block(output, skip_size)
            skips.append(skip)

        return torch.stack(skips)



class OutNet(nn.Module):
    def __init__(self, channels):
        super(OutNet, self).__init__()

        self.relu = nn.ReLU()

        self.layer1 = nn.Conv1d(channels, channels, 1)
        self.layer2 = nn.Conv1d(channels, channels, 1)

        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        output = self.relu(x)
        output = self.layer1(x)
        output = self.relu(x)
        output = self.layer2(x)
        return output
