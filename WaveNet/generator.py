import torch
import numpy as np
import os
from soundfile import write as write_wav

from WaveNet.Wavenet import *
import WaveNet.data as data

class Generator:
    def __init__(self, model, model_dir, step, sample_rate, seed_path, out_path, sample_size):
        """
        Args:
            model      (WaveNet) : Trained WaveNet model
            model_dir     (path) : Path to the model's saved parameters
            step           (int) : The number of the model to load
            sample_rate    (int) : Sample rate at which to generate
            seed_path     (path) : Path to the seed file
            out_path      (path) : File path of output file
            sample_size    (int) : Number of samples to generate
        """
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.seed = seed_path
        self.out_path = out_path

        self.wavenet = model
        self.wavenet.load(model_dir, step)

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _make_seed(self, audio):
        audio = np.pad([audio], [[0, 0], [self.wavenet.receptive_field, 0], [0, 0]], 'constant')

        if self.sample_size:
            seed = audio[:, :self.sample_size, :]
        else:
            seed = audio[:, :self.wavenet_receptive_field * 2, :]

        return seed

    def _get_seed_from_audio(self, filepath):
        audio = data.load(filepath, self.sample_rate)
        audio_length = len(audio)

        audio = data.mu_law_encode(audio, self.wavenet.in_channels)
        audio = data.one_hot_encode(audio, self.wavenet.in_channels)

        seed = self._make_seed(audio)

        return self._variable(seed), audio_length

    def _save_to_audio_file(self, output):
        output = output[0].cpu().data.numpy()
        output = data.one_hot_decode(output, axis=1)
        waveform = data.mu_law_decode(output, self.wavenet.in_channels)

        write_wav(self.out_path, waveform, self.sample_rate)
        print('Saved wav file at {}'.format(self.out_path))


    def generate(self):
        outputs = []
        inputs, audio_length = self._get_seed_from_audio(self.seed)

        while True:
            new = self.wavenet.generate(inputs)

            outputs = torch.cat((outputs, new), dim=1) if len(outputs) else new

            print('{0}/{1} samples are generated.'.format(len(outputs[0]), audio_length))

            if len(outputs[0] >= audio_length):
                break

            inputs = torch.cat((inputs[:, :-len(new[0]), :], new), dim=1)

        outputs = outputs[:, :audio_length, :]

        self._save_to_audio_file(outputs)
