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
        self.ims_data = self.load_ims_data()
        self.dme_data = self.load_dme_data()

        # norm. - https://datascience.stackexchange.com/questions/5885/how-to-scale-an-array-of-signed-integers-to-range-from-0-to-1
        self.ims_data = (self.ims_data - self.ims_data.min()) / (self.ims_data.max() - self.ims_data.min())
        self.dme_data = (self.dme_data - self.dme_data.min()) / (self.dme_data.max() - self.dme_data.min())

        # set dim.
        self.m, self.k_m = self.dme_data.shape
        self.n, self.k_n = self.ims_data.shape

        print("X dim: {} - has {} links, {} samples each, {} days \n"
              "Y dim: {} - has {} gauges, {} samples each, {} days"
            .format(
            self.dme_data.shape, self.m, self.k_m, config.coverage,
            self.ims_data.shape, self.n, self.k_n, config.coverage)
        )

    def get_entry(self, arr):
        i = 0
        while (not ('name' in arr[i] and arr[i]['name'] == 'Rain')):
            i += 1

        return arr[i]

    def load_ims_data(self):
        if not config.ims_pre_load_data:

            # 10[min] x 6 is an hour
            ims_matrix = np.empty((1, 6 * 24 * config.coverage))
            for index, station_folder in enumerate(os.listdir(config.ims_root_files)):
                print("extractor: now processing gauge: {}".format(station_folder))
                try:
                    df = pd.read_csv(config.ims_root_files + '/' + station_folder + '/' + 'data.csv')
                    ims_vec = np.empty(1)
                    for row_time, row in zip(list(df.datetime), list(df.channels)):
                        ims_vec = np.vstack((ims_vec, np.array([self.get_entry(ast.literal_eval(row))['value']])))

                    ims_vec = ims_vec[1:].T
                    try:
                        ims_matrix = np.vstack((ims_matrix, ims_vec))
                    except ValueError:
                        print("problem with stacking gague {}".format(station_folder))

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_folder))

            ims_matrix = ims_matrix[1:]

            if not os.path.isdir(config.ims_root_values + '/' + config.date_str_rep):
                os.makedirs(config.ims_root_values + '/' + config.date_str_rep)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'wb') as f:
                pickle.dump(ims_matrix, f)

            return ims_matrix

        else:
            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'rb') as f:
                ims_matrix = pickle.load(f)

            return ims_matrix

    def load_dme_data(self):
        if not config.dme_pre_load_data:

            # 15[min] x 4 is an hour
            dme_matrix = np.empty((1, 4 * 24 * config.coverage))

            for link in os.listdir(config.dme_root_files):

                link_name = link.split('_')[0]
                link_type = config.dme_scrape_config['link_objects']['measurement_type']

                print("extractor: now processing link: {} of type: {}".format(link_name, link_type))

                df = pd.read_csv(config.dme_root_files + '/' + link)

                #TODO need to filter out rows that don't have row_interval==15
                time_value = config.date_datetime_rep


                if len(list(df['rain'])) == 4 * 24 * config.coverage:
                    dme_matrix = np.vstack((dme_matrix, list(df['rain'])))

                else: # fill with nan for missing data

                    dme_vec = np.empty(1)

                    for row_value, row_time, row_interval in zip(list(df['rain']), list(df.Time), list(df.Interval)):

                        # fill with rain data
                        if dme_vec[1:].size < 4 * 24 * config.coverage:

                            if time_value>dt.strptime(row_time, "%Y-%m-%d %H:%M:%S"):
                                raise ValueError('Something went wrong, extractor time missmatch')

                            #fill with nan
                            while time_value < dt.strptime(row_time, "%Y-%m-%d %H:%M:%S"):
                                dme_vec = np.hstack((dme_vec, np.nan))
                                time_value = time_value + dt_delta(minutes=15)

                            # fill with actual dme data
                            if row_value != np.nan:
                                dme_vec = np.hstack((dme_vec, row_value))
                                time_value = time_value + dt_delta(minutes=15)

                            else:
                                dme_vec = np.hstack((dme_vec, np.nan))
                                time_value = time_value + dt_delta(minutes=15)



                    while len(dme_vec) <= 4 * 24 * config.coverage:
                        dme_vec = np.hstack((dme_vec, np.nan))

                    dme_matrix = np.vstack((dme_matrix, dme_vec[1:].T))


            dme_matrix = dme_matrix[1:]

            if not os.path.isdir(config.dme_root_values + '/' + config.date_str_rep):
                os.makedirs(config.dme_root_values + '/' + config.date_str_rep)

            with open(config.dme_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'wb') as f:
                pickle.dump(dme_matrix, f)

            return dme_matrix

        else:
            with open(config.dme_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'rb') as f:
                dme_matrix = pickle.load(f)
            return dme_matrix


if __name__ == "__main__":
    Extractor()
