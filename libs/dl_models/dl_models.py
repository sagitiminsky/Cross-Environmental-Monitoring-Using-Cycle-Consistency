import config
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.callbacks import Callback
import numpy as np
import wandb
from wandb.keras import WandbCallback
import pandas as pd
import os
import ast
import pickle


class DL_Models:
    def __init__(self):
        self.generator = self.pretrain_generator()
        self.critic = self.pretrain_critic()


    def pretrain_generator(self):
        run = wandb.init()
        wandb_config = run.config
        wandb_config.encoding_dim = config.generator_encoding_dim
        wandb_config.epochs = config.generator_epoches

        # load dme and ims data
        x_test = self.load_ims_data()
        x_train = self.load_dme_data()

        # norm.
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(config.encoding_dim, activation='relu'))
        model.add(Dense(28 * 28, activation='sigmoid'))
        model.add(Reshape((28, 28)))
        model.compile(optimizer='adam', loss='mse')

        # For visualization
        class Images(Callback):
            def on_epoch_end(self, epoch, logs):
                indices = np.random.randint(self.validation_data[0].shape[0], size=8)
                test_data = self.validation_data[0][indices]
                pred_data = self.model.predict(test_data)
                run.history.row.update({
                    "examples": [
                        wandb.Image(np.hstack([data, pred_data[i]]), caption=str(i))
                        for i, data in enumerate(test_data)]
                })

        model.fit(x_train, x_train,
                  epochs=config.epochs,
                  validation_data=(x_test, x_test),
                  callbacks=[Images(), WandbCallback()])

        model.save('auto.h5')

    def load_ims_data(self):

        if not config.load_previous:
            x_test = []
            for index, station_folder in enumerate(os.listdir(config.ims_root_files)):

                print("now processing {}".format(station_folder))
                try:
                    df = pd.read_csv(config.ims_root_files + '/' + station_folder + '/' + 'data.csv')
                    x_test.append([ast.literal_eval(row)[0]['value'] for row in list(df.channels)])

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_folder))

            with open('parrot.pkl', 'wb') as f:
                pickle.dump(mylist, f)

            return x_test



    def pretrain_critic(self):
        pass


if __name__ == "__main__":
    DL_Models()
