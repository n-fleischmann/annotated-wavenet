"""
Extract audio & Mu law compand
"""
import os
import librosa
import numpy as np
import torch
import torch.utils.data as d

def load(filename, sample_rate=16000, trim=2048):
    '''
    filename (string): Path to audio file
    sample_rate (int): Sample rate to import file with
    Trim (int): number of samples in each clip, 0 returns whole file

    returns:
        audio     (NP array) : shape (trim,)
          OR
        raw_audio (NP array) : shape (n,)
    '''
    # Load in the raw audio using librosa and reshape to remove extra dimension
    # Underscore is the sample rate, which we already know
    raw_audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    raw_audio = raw_audio.reshape(-1, 1)

    # If requested, trim the clips of the specified number of samples
    if trim > 0:
        audio, _ = librosa.effects.trim(raw_audio, frame_length=trim)
        return audio

    return raw_audio


def one_hot_encode(data, channels=256):
    '''
    Creates a one-hot encoding of the 1D input data

    returns np array of size (data.size, channels)
    '''
    one_hot = np.zeros((data.size, channels), dtype=float)
    # Make the value one at the the column corresponding to the row value
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot


def one_hot_decode(data, axis=1):
    '''
    Decodes a one-hot encoding

    returns a 1D array
    '''
    return np.argmax(data, axis=axis)


def mu_law_encode(waveform, channels=256):
    '''
    Quantize waveform amplitudes
    '''

    # create an array of equally spaced points between -1 and 1
    lin_space = np.linspace(-1, 1, channels)

    # Apply the formula
    channels = float(channels)
    quantized = np.sign(waveform) * np.log(1 + (channels - 1) * np.abs(waveform)) / np.log(channels)

    # Discretize and return
    return np.digitize(quantized, lin_space) - 1



def mu_law_decode(data, channels=256):
    '''
    Recovers the waveform from discretized data
    '''
    channels = float(channels)

    exp = -1 + (data / channels) *2.0
    return np.sign(exp) * (np.exp(np.abs(exp) * np.log(channels)) - 1) / (channels - 1)



class CustomDataset(d.Dataset):
    def __init__(self, dir, sr=16000, channels=256, trim=2048):
        '''
        dir : Path to directory of audio files
        sr  : Sample rate of audio files
        channels : Number of quantization channels
        trim : Number of samples in each segment, 0 is no trim
        '''
        super(CustomDataset, self).__init__()

        self.channels = channels
        self.sample_rate = sr
        self.trim = trim

        self.path = dir
        self.filenames = [filename for filename in sorted(os.listdir(dir))]

    def __getitem__(self, idx):
        file = os.path.join(self.path, self.filenames[idx])

        audio = load(file, self.sample_rate, self.trim)

        companded = mu_law_encode(audio, self.channels)

        return one_hot_encode(companded, self.channels)

    def __len__(self):
        return len(self.filenames)


class WNLoader(d.DataLoader):
    def __init__(self, dir, receptive_field,
                sample_size=0, sr=16000, companding_channels=256,
                batch_size=1, shuffle=True):
        '''
        Loads data for WaveNet

        dir                 : Directory of audio files
        receiptive_field    : length of the receptive field of the model
        sample_size         : Number of samples in one input instance,
                                must be bigger than the reciptive field
        sr                  : Sample rate of training audio
        companding_channels : number of channels to Mu law compand to
        batch_size          : Number of instances outputed at once
        shuffle             : Are the instances in random order?
        '''
        dataset = CustomDataset(dir, sr, companding_channels)

        super(WNLoader, self).__init__(dataset, batch_size, shuffle)

        if sample_size <= receptive_field:
            raise Exception("Sample size {} must be larger than receptive field {}".format(sample_size, receiptive_field))

        self.sample_size = sample_size
        self.receptive_field = receptive_field

        self.collate_fn = self._collate_fn

    def _sample_size(self, audio):
        return min(self.sample_size, len(audio[0]))

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        return torch.autograd.Variable(tensor)

    def _collate_fn(self, audio):
        # First pad the audio along the timestep dimension to make sure its longer
        # than the required sample size, it gets cut down later
        audio = np.pad(audio, [[0, 0], [self.receptive_field, 0], [0, 0]], 'constant')

        if self.sample_size:
            # If a sample size greater than 0 is given, break the file into
            # sections that are that long
            sample_size = self._sample_size(audio)

            # This while loop breaks the waveform up into chunks and returns them
            # one at a time
            while sample_size > self.receptive_field:
                inputs = audio[:, :sample_size, :]
                targets = audio[:, self.receptive_field:sample_size, :]

                yield self._variable(inputs), self._variable(one_hot_decode(targets, 2))

                audio = audio[:, sample_size - self.receptive_field:, :]
                sample_size = self._sample_size(audio)
        else:
            # if the sample size is zero or None or False, return the whole file
            targets = audio[:, self.receiptive_field:, :]
            return self._variable(audio), self._variable(one_hot_decode(targets, 2))
