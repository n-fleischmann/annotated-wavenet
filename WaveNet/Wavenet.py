import torch
from torch import nn
import time
import os

from WaveNet.model_parts import *
from WaveNet.exceptions import InputSizeError

class WaveNet(nn.Module):
    def __init__(self, num_blocks, stack_size, in_channels, residual_channels, lr=0.002):
        """
        Params:
            num_blocks (int)        : The number of blocks in the stack
            stack_size (int)        : The number of dilations in each block
            in_channels (int)       : number channels for input data,
                                      should be the value of mu in companding
            residual_channels (int) : Number of internal channels of each block
            lr (float)              : Learning rate of the optimizer
        """
        super(WaveNet, self).__init__()

        self.in_channels = in_channels

        self.start_conv = CausalConv(in_channels, residual_channels)
        self.stack = BlockStack(num_blocks, stack_size, residual_channels, in_channels)
        self.outnet = OutNet(in_channels)

        self.receptive_field = self.calc_receptive_field(num_blocks, stack_size)

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

    @staticmethod
    def calc_receptive_field(num_blocks, stack_size):
        return sum([2**i for i in range(0, stack_size)]) * num_blocks

    @staticmethod
    def _loss():
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _output_size(self, x):
        output_size = int(x.shape[1]) - self.receptive_field

        if output_size < 1:
            raise InputSizeError(int(x.shape[1]), self.receptive_field, output_size)

        return output_size

    def forward(self, x, verbose=False):
        output = x.transpose(1, 2)

        output = self.start_conv(output)

        if verbose: print('Dilating:')

        skips = self.stack(output, self._output_size(x))

        output = torch.sum(skips, dim=0)

        output = self.outnet(output)

        return output.transpose(1, 2).contiguous()

    def train(self, inputs, targets, verbose=False, timer=False):
        if timer: start_time = time.time()

        # Train one time
        outputs = self.forward(inputs, verbose=verbose)

        loss = self.loss(outputs.view(-1, self.in_channels),
                        targets.long().view(-1))

        if timer:
            print('\t\tLoss: {:.6f}'.format(loss.item()) \
                    + '\tTime: {:.2f} seconds'.format(time.time() - start_time))
        else:
            print('\t\tLoss: {}'.format(loss.item()))

        self.optimizer.zero_grad()

        if verbose: print('Backpropagating...')
        loss.backward()
        if verbose: print('Optimizing...')
        self.optimizer.step()

        return loss.item()


    def generate(self, inputs):
        # Generate 1 time
        return self.forward(inputs)


    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))


    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.load_state_dict(torch.load(model_path))


    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.state_dict(), model_path)
