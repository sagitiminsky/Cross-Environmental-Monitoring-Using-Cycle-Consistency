import config
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback
import pandas as pd
import os
import ast
import pickle
import numpy as np
from .save_and_load import Save_Or_Load_Model



class DL_Models:
    def __init__(self):
        self.generator = self.pre_train_generator()
        self.critic = self.pre_train_critic()

    def pre_train_generator(self):
        run = wandb.init()
        wandb_config = run.config
        wandb_config.encoding_dim = config.generator_encoding_dim
        wandb_config.epochs = config.generator_epoches

        # load dme and ims data
        dme_data = self.load_dme_data()
        ims_data = self.load_ims_data()

        # norm.
        dme_data = np.array(dme_data.astype('float32') / np.max(dme_data))
        ims_data = np.array(ims_data.astype('float32') / np.max(ims_data))

        # validate dim
        if dme_data.shape[1] != ims_data.shape[1]:
            raise ValueError(
                "dimension of k in x_train and x_test do not feet {} {}".format(dme_data.shape[1], ims_data.shape[1]))

        # set dim.
        m, k = dme_data.shape
        n, k = ims_data.shape

        model = Sequential()
        model.add(Dense(n))
        model.add(Dense(config.generator_encoding_dim, activation='relu'))
        model.add(Dense(m, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse')

        model.fit(dme_data.T[:int(len(dme_data) * 0.8)], ims_data.T[:int(len(ims_data) * 0.8)],
                  epochs=config.generator_epoches,
                  validation_data=(dme_data[int(len(dme_data) * 0.8):], ims_data[int(len(dme_data) * 0.8):]))


        model_manager=Save_Or_Load_Model()
        model_manager.save_onnx_model(model)
        model_manager.save_model(model)

    def load_ims_data(self):

        if not config.ims_pre_load_data:
            x_test = []
            for index, station_folder in enumerate(os.listdir(config.ims_root_files)):

                print("now processing {}".format(station_folder))
                try:
                    df = pd.read_csv(config.ims_root_files + '/' + station_folder + '/' + 'data.csv')
                    x_test.append([ast.literal_eval(row)[0]['value'] for row in list(df.channels)])

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_folder))

            with open(config.ims_root_values + '/' + 'ims_values.pkl', 'wb') as f:
                pickle.dump(x_test, f)

            return x_test

        else:
            with open(config.ims_root_values + '/' + 'ims_values.pkl', 'rb') as f:
                x_test = pickle.load(f)
            return x_test

    def pre_train_critic(self):
        raise (NotImplemented)





if __name__ == "__main__":
    DL_Models()
