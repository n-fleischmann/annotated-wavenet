from WaveNet.Wavenet import WaveNet
from WaveNet.generator import Generator


if __name__ == '__main__':
        model = WaveNet(5, 10, 256, 512)
        '''
        for save_step in [2, 50, 100, 150, 200, 250, 300, 317]:
            outpath = './generator_output/gen_{}.wav'.format(save_step)
            generator = Generator(model, './model_saves',
                save_step, 16000, './data/helloworld.wav', outpath, 100000)
            generator.generate()'''

        generator = Generator(model, './model_saves',
            317, 16000, './data/arctic_a0001.wav', './generator_output/gen_counter.wav', 100000)
        generator.generate()
