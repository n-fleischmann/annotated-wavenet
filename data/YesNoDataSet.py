import torch
import torchaudio
import random
from itertools import *
from torch.utils.data import IterableDataset

"""Adapted from
https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
"""

class YesNoWaveNetDataSet(IterableDataset):
    """YesNo, adapted for WaveNet"""

    _ext_audio = '.wav'

    def __init__(self, root, download=False, channels=10):

        self._raw = torchaudio.datasets.YESNO(root=root, download=download)
        self.channels = channels
        self.waveforms = [item[0][0] for item in self._raw]

        assert len(self.waveforms) % self.channels == 0, "Channels should divide in " + len(self._ids)



    @property
    def shuffled_data_list(self):
        return random.sample(self.waveforms, len(self.waveforms))

    def process_data(self,data):
        for x in data:
            yield torch.tensor([x])

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.channels)])

    def __iter__(self):
        self._n = 0
        self._internal_iter = self.get_streams()
        self._next = torch.cat(tuple(next(self._internal_iter)), 0)
        return self

    def __next__(self):
        if self._n < len(self):
            self._n += 1
            ret = self._next
            self._next = torch.cat(tuple(next(self._internal_iter)), 0)
            return ret, self._next
        else:
            raise StopIteration


    def __len__(self):
        l = 0
        for item in self.waveforms:
            l += len(item)
        return int(l / self.channels)
