import WaveNet.data as data

class Trainer:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir

    def train(self, dir, sample_size, epochs=None, verbose=False, timer=False):
        '''
        Trains the model from files in the given directory

        Args:
            dir (str)         : Path to training files
            sample_size (int) : Number of samples in each file section, None for whole file
            epochs (int)      : Number of epochs to train, None to run until stopped
            verbose (bool)    : Should this print extra info about dilating and optimizing?
            timer (bool)      : Should this print extra info about the time taken to calculate?

        Returns: List of losses
        '''
        loader = data.WNLoader(dir, self.model.receptive_field, sample_size=sample_size)

        counter = 1

        while True:
            if epochs is not None and counter > epochs:
                break

            print('='*10, 'Epoch {}'.format(counter), '='*10)
            file_counter, epoch_losses = 1, []

            # Loop through the files
            for file in loader:
                print('\tFile {}'.format(file_counter))

                # Loop through the sections of each file
                for inputs, targets in file:
                    loss = self.model.train(inputs, targets, verbose=verbose, timer=timer)
                    epoch_losses.append(loss)
                file_counter += 1
                self.model.save(self.save_dir, step=file_counter)
            counter += 1

        return epoch_losses
