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
from sklearn.model_selection import train_test_split


class DL_Models:
    def __init__(self):
        # load dme and ims data
        self.dme_data = self.load_dme_data()
        self.ims_data = self.load_ims_data()
        self.dme_max_value = np.max(self.dme_data)
        self.ims_max_value = np.max(self.ims_data)

        # norm.
        self.dme_data = np.array(self.dme_data.astype('float32') / self.dme_max_value)
        self.ims_data = np.array(self.ims_data.astype('float32') / self.ims_max_value)

        # validate dim
        if self.dme_data.shape[1] != self.ims_data.shape[1]:
            raise ValueError(
                "dimension of k in x_train and x_test do not feet {} {}".format(self.dme_data.shape[1],
                                                                                self.ims_data.shape[1]))

        # set dim.
        self.m, self.k = self.dme_data.shape
        self.n, self.k = self.ims_data.shape

        self.generator, self.generated_features = self.pre_train_generator()
        self.critic = self.pre_train_critic()

    def pre_train_generator(self):
        model_manager = Save_Or_Load_Model()

        if not config.load_pre_trained_generator:
            run = wandb.init()
            wandb_config = run.config
            wandb_config.encoding_dim = config.generator_encoding_dim
            wandb_config.epochs = config.generator_epoches

            model = Sequential()
            model.add(Dense(self.n))
            model.add(Dense(config.generator_encoding_dim, activation='relu'))
            model.add(Dense(self.m, activation='sigmoid'))
            model.compile(optimizer='adam', loss='mse')

            model.fit(self.dme_data.T[:int(len(self.dme_data) * 0.8)], self.ims_data.T[:int(len(self.ims_data) * 0.8)],
                      epochs=config.generator_epoches,
                      validation_data=(
                          self.dme_data[int(len(self.dme_data) * 0.8):], self.ims_data[int(len(self.dme_data) * 0.8):]))

            model_manager.save_onnx_model(model,config.generator_onnx_path)
            model_manager.save_model(model,config.generator_path)
        else:
            model = model_manager.load_model(config.generator_path)

        return model, [model.predict(x) for x in self.dme_data.T]

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

    def load_dme_data(self):
        raise NotImplemented

    def pre_train_critic(self):
        model_manager = Save_Or_Load_Model()

        if not config.load_pre_trained_critic:

            # logging code
            run = wandb.init()
            wandb_config = run.config

            wandb_config.epochs = config.critic_epochs

            # load data
            X_train, y_train, X_test, y_test = train_test_split(
                np.vstack((self.generated_features, self.ims_data)),
                [0] * len(self.generated_features) + [1] * len(self.ims_data), test_size=0.33, random_state=42)

            is_GRA_train = y_train == 0
            is_GRA_test = y_test == 0
            labels = ["GRA", "RA"]

            # create model
            model = Sequential()
            model.add(Dense(self.k))
            model.add(Dense(1), activation='sigmoid')
            model.compile(loss='mse', optimizer='adam',
                          metrics=['accuracy'])

            # Fit the model
            model.fit(X_train, is_GRA_train, epochs=config.critic_epochs, validation_data=(X_test, is_GRA_test),
                      callbacks=[WandbCallback(labels=labels)])

            model_manager.save_onnx_model(model,config.critic_onnx_path)
            model_manager.save_model(model,config.critic_path)

        else:
            model = model_manager.load_model(config.critic_path)

        return model

if __name__ == "__main__":
    DL_Models()
