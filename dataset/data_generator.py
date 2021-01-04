import numpy as np
from tensorflow.keras.utils import Sequence

from utils.loader import pkl_loader


class DataGenerator(Sequence):

    def __init__(self, pkl_fp, batch_size, shuffle=False):
        super(DataGenerator, self).__init__()
        pkl_data = pkl_loader(pkl_fp)
        if len(pkl_data.shape) == 2:
            pkl_data = pkl_data[:, :, np.newaxis]
        self.pkl_data = pkl_data
        self.batch_size = batch_size
        self.row_idx = np.arange(self.pkl_data.shape[0])
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.row_idx)

    def __len__(self):
        return int(np.floor(self.pkl_data.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.pkl_data[self.row_idx[idx * self.batch_size:(idx + 1) * self.batch_size], :-1]

        batch_y = self.pkl_data[self.row_idx[idx * self.batch_size:(idx + 1) * self.batch_size], -1]
        return batch_x, batch_y
