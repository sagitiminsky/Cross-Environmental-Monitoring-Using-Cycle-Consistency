import CellEnMon.config as config
from CellEnMon.libs.preprocessing.extractor import Extractor
from sklearn.model_selection import train_test_split
import uuid
import numpy as np


class Augmentor(Extractor):
    def __init__(self):
        super().__init__()
        self.augmentation(self.dme_data_tiled, self.ims_data_tiled)

    def augmentation(self, A, B):
        for augmentation in config.augmentations:
            if augmentation == 0:
                trainA, testA, trainB, testB = train_test_split(A.T, B.T, test_size=0.1, random_state=42)
                trainA, testA, trainB, testB = trainA.T, testA.T, trainB.T, testB.T
            else:
                raise NotImplemented('This augmentation type is not implemented {}'.format(augmentation))

        d = {'trainA': trainA, 'testA': testA, 'trainB': trainB, 'testB': testB}
        for tag in d:
            save_dir = './libs/relics/datasets/dme2ims/' + tag

            for col in d[tag].T:
                np.save(save_dir + '/' + str(uuid.uuid4()), col)


if __name__ == "__main__":
    Augmentor()
