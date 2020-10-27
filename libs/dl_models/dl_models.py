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
from libs.dl_models.save_and_load import Save_Or_Load_Model
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
from datetime import timedelta as dt_delta
from libs.power_law.power_law import PowerLaw


class DL_Models:
    def __init__(self):
        # self.ims_data = np.array(self.load_ims_data())
        self.dme_data = np.array(self.load_dme_data())
        self.dme_max_value = np.max(self.dme_data)
        self.dme_min_value = np.min(self.dme_data)
        self.ims_max_value = np.max(self.ims_data)
        self.ims_min_value = np.min(self.ims_data)


        # norm. - https://datascience.stackexchange.com/questions/5885/how-to-scale-an-array-of-signed-integers-to-range-from-0-to-1
        self.dme_data = (np.array(self.dme_data) - self.dme_min_value) / (self.dme_max_value - self.dme_min_value)
        self.ims_data = (np.array(self.ims_data) - self.ims_min_value) / (self.ims_max_value - self.ims_min_value)

        # set dim.
        self.m, self.k_m = self.dme_data.shape
        self.n, self.k_n = self.ims_data.shape

        print("X dim: {} - has {} links, {} samples each \nY dim: {} - has {} gauges, {} samples each".format(
            self.dme_data.shape, self.m, self.k_m, self.ims_data.shape, self.n, self.k_n))

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

            model_manager.save_onnx_model(model, config.generator_onnx_path)
            model_manager.save_model(model, config.generator_path)
        else:
            model = model_manager.load_model(config.generator_path)

        return model, [model.predict(x) for x in self.dme_data.T]

    def load_ims_data(self):
        if not config.ims_pre_load_data:
            x_test = np.empty((1, 6 * 24 * config.coverage))
            for index, station_folder in enumerate(os.listdir(config.ims_root_files)):
                print("now processing gauge: {}".format(station_folder))
                try:
                    df = pd.read_csv(config.ims_root_files + '/' + station_folder + '/' + 'data.csv')
                    values = np.empty(1)
                    for row_time, row in zip(list(df.datetime), list(df.channels)):
                        values = np.vstack((values, np.array([ast.literal_eval(row)[0]['value']])))
                    values = values[1:].T
                    x_test = np.vstack((x_test, values))

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_folder))

            x_test = x_test[1:]

            with open(config.ims_root_values + '/' + 'ims_values.pkl', 'wb') as f:
                pickle.dump(x_test, f)

            return x_test

        else:
            with open(config.ims_root_values + '/' + 'ims_values.pkl', 'rb') as f:
                x_test = pickle.load(f)
            return x_test

    def load_dme_data(self):
        if not config.dme_pre_load_data:

            """
            The following dme_matrix is constructed as follows:
            each element consist of link data, refrence can be found at link_matrix
            """
            dme_matrix=[]

            for link in os.listdir(config.dme_root_files):

                """
                The following link_matrix is constructed as follows:
                each row consist of 4*96 values the represent a day's worth of mesasurement of a link
                """
                link_matrix=[]

                link_name = link.split('_')[0]
                link_type = config.dme_scrape_config['link_objects']['measurement_type']
                link_frequency = int(int(link.split('_')[2])*10**-3) ## link frequency needs to be in Ghz and in metadata it is in MHz
                link_polarization= link.split('_')[4]
                link_L = link.split('_')[6]

                print("now processing link: {} of type: {}".format(link_name, link_type))

                """
                The following part will take into account the metadata and input it to the PowerLaw
                """

                power_law=PowerLaw(frequency=link_frequency,polarization=link_polarization,L=link_L)



                df = pd.read_csv(config.dme_root_files + '/' + link)
                init_date = config.dme_scrape_config['link_objects']['date']['value']
                time_value = dt.strptime(f"{init_date['yyyy']}-{init_date['mm']}-{init_date['dd']} 00:00:00", "%Y-%m-%d %H:%M:%S") + dt_delta(days=1)



                dme_vector = []
                try:
                    for row_value, row_time, row_interval in zip(list(df['RFInputPower']),list(df.Time), list(df.Interval)):

                        #fill with nan for missing data
                        while time_value < dt.strptime(row_time, "%Y-%m-%d %H:%M:%S"):
                            dme_vector.append(np.nan)
                            time_value = time_value + dt_delta(minutes=15)
                            if len(dme_vector) == 4 * 24:
                                link_matrix.append(dme_vector)
                                dme_vector = []


                        if row_interval != 24:
                            if row_value!=np.nan:
                                dme_vector.append(power_law.basic_attinuation_to_rain(float(row_value)))
                            else:
                                dme_vector.append(np.nan)
                            time_value = dt.strptime(row_time, "%Y-%m-%d %H:%M:%S") + dt_delta(minutes=15)
                            if len(dme_vector) == 4 * 24:
                                link_matrix.append(dme_vector)
                                dme_vector = []

                    dme_matrix.append(link_matrix)
                except KeyError:
                    print("now processing link: {} of type: {} was not successful".format(link_name, link_type))


            with open(config.dme_root_values + '/' + 'dme_values.pkl', 'wb') as f:
                pickle.dump(dme_matrix, f)

            return dme_matrix

        else:
            with open(config.dme_root_values + '/' + 'dme_values.pkl', 'rb') as f:
                dme_matrix = pickle.load(f)
            return dme_matrix

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

            model_manager.save_onnx_model(model, config.critic_onnx_path)
            model_manager.save_model(model, config.critic_path)

        else:
            model = model_manager.load_model(config.critic_path)

        return model


if __name__ == "__main__":
    DL_Models()
