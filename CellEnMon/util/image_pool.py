import random
import torch


class SignalPool():
    """This class implements an signal buffer that stores previously generated signals.

    This buffer enables us to update discriminators using a history of generated signals
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the signalPool class

        Parameters:
            pool_size (int) -- the size of signal buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.signals = []

    def query(self, signals):
        """Return an signal from the pool.

        Parameters:
            signals: the latest generated sigmal from the generator

        Returns signals from the buffer.

        By 50/100, the buffer will return input signals.
        By 50/100, the buffer will return signals previously stored in the buffer,
        and insert the current signals to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return signals
        return_signals = []
        for signal in signals:
            signal = torch.unsqueeze(signal.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current signals to the buffer
                self.num_imgs = self.num_imgs + 1
                self.signals.append(signal)
                return_signals.append(signal)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored signal, and insert the current signal into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.signals[random_id].clone()
                    self.signals[random_id] = signal
                    return_signals.append(tmp)
                else:       # by another 50% chance, the buffer will return the current signal
                    return_signals.append(signal)
        return_signals = torch.cat(return_signals, 0)   # collect all the signals and return
        return return_signals
