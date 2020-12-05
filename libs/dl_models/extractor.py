import config
import pickle
import numpy as np
import pandas as pd
import os
import ast
from datetime import datetime as dt
from datetime import timedelta as dt_delta

class Extractor:
    def __init__(self):
        self.ims_data,self.ims_max_value, self.ims_min_value = self.load_ims_data()
        self.dme_data = np.array(self.load_dme_data())
        self.dme_max_value, self.dme_min_value = np.max(self.dme_data), np.min(self.dme_data)

        # norm. - https://datascience.stackexchange.com/questions/5885/how-to-scale-an-array-of-signed-integers-to-range-from-0-to-1
        self.dme_data = (np.array(self.dme_data) - self.dme_min_value) / (self.dme_max_value - self.dme_min_value)
        self.ims_data = (np.array(self.ims_data) - self.ims_min_value) / (self.ims_max_value - self.ims_min_value)

        # set dim.
        self.m, self.k_m = self.dme_data.shape
        self.n, self.k_n = self.ims_data.shape

        print("X dim: {} - has {} links, {} samples each \nY dim: {} - has {} gauges, {} samples each".format(
            self.dme_data.shape, self.m, self.k_m, self.ims_data.shape, self.n, self.k_n))

    def load_ims_data(self):
        if not config.ims_pre_load_data:

            min_ims = 10000
            max_ims = 0

            # 10[min] x 6 is an hour
            x_test = np.empty((1, 6 * 24 * config.coverage))
            for index, station_folder in enumerate(os.listdir(config.ims_root_files)):
                print("now processing gauge: {}".format(station_folder))
                try:
                    df = pd.read_csv(config.ims_root_files + '/' + station_folder + '/' + 'data.csv')
                    values = np.empty(1)
                    for row_time, row in zip(list(df.datetime), list(df.channels)):
                        values = np.vstack((values, np.array([ast.literal_eval(row)[0]['value']])))

                        if max_ims < np.max(values):
                            max_ims = np.max(values)
                        if min_ims > np.min(values):
                            min_ims = np.min(values)

                    values = values[1:].T
                    try:
                        x_test = np.vstack((x_test, values))
                    except ValueError:
                        print("problem with stacking gague {}".format(station_folder))

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_folder))

            x_test = x_test[1:]

            if not os.path.isdir(config.ims_root_values + '/' + config.date_str_rep):
                os.makedirs(config.ims_root_values + '/' + config.date_str_rep)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'wb') as f:
                pickle.dump(x_test, f)
            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + "metadata.txt", 'w') as file:
                file.write('min: {}\n'.format(min_ims))
                file.write('max: {}\n'.format(max_ims))

            return x_test,min_ims,max_ims

        else:
            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'rb') as f:
                x_test = pickle.load(f)
            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + "metadata.txt", 'w') as file:
                min_ims=float(file.readline().split(' ')[1])
                max_ims = float(file.readline().split(' ')[1])

            return x_test,min_ims,max_ims

    def load_dme_data(self):
        if not config.dme_pre_load_data:

            """
            The following dme_matrix is constructed as follows:
            each element consist of link data, refrence can be found at link_matrix
            """
            dme_matrix = []

            for link in os.listdir(config.dme_root_files):

                """
                The following link_matrix is constructed as follows:
                each row consist of 4*96 values the represent a day's worth of mesasurement of a link
                """
                link_matrix = []

                link_name = link.split('_')[0]
                link_type = config.dme_scrape_config['link_objects']['measurement_type']

                print("now processing link: {} of type: {}".format(link_name, link_type))

                df = pd.read_csv(config.dme_root_files + '/' + link)
                init_date = config.dme_scrape_config['link_objects']['date']['value']
                time_value = dt.strptime(f"{init_date['yyyy']}-{init_date['mm']}-{init_date['dd']} 00:00:00",
                                         "%Y-%m-%d %H:%M:%S") + dt_delta(days=1)

                dme_vector = []
                try:
                    for row_value, row_time, row_interval in zip(list(df['rain']), list(df.Time), list(df.Interval)):

                        # fill with nan for missing data
                        while time_value < dt.strptime(row_time, "%Y-%m-%d %H:%M:%S"):
                            dme_vector.append(np.nan)
                            time_value = time_value + dt_delta(minutes=15)
                            if len(dme_vector) == 4 * 24:
                                link_matrix.append(dme_vector)
                                dme_vector = []

                        if row_interval != 24:
                            if row_value != np.nan:
                                dme_vector.append(row_value)
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



if __name__=="__main__":
    Extractor()