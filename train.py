import WaveNet.Wavenet as Wavenet
import WaveNet.trainer as trainer
import warnings
from torchaudio.datasets import CMUARCTIC

if __name__ == '__main__':
    # The sox backend is deprecated, but we can ignore that for now
    warnings.filterwarnings("ignore")

    # download data
    CMUARCTIC(root='./data/', url='bdl', download=True)

    # Initialize and train the model
    model = Wavenet.WaveNet(5, 12, 256, 512)
    print('Receptive Field:', model.receptive_field)
    trainer = trainer.Trainer(model, './model_saves/')
    trainer.train('./data/ARCTIC/cmu_us_bdl_arctic/wav/', 25000, epochs=1, timer=True)
