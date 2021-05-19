import CellEnMon.config as config
import pickle
import numpy as np
import pandas as pd
import os
import ast
import re
from datetime import datetime as dt
from datetime import timedelta as dt_delta


class Extractor:
    def __init__(self):
        self.ims_data, self.ims_order = self.load_ims_data()  # 1 day is 144 = 24*6 samples
        self.dme_data, self.dme_order = self.load_dme_data()  # 1 day is 96 = 24*4 samples

        ############
        ### norm. - https://datascience.stackexchange.com/questions/5885/how-to-scale-an-array-of-signed-integers-to-range-from-0-to-1
        ############

        self.ims_data = self.ims_data.astype(np.float)
        self.dme_data = self.dme_data.astype(np.float)

        # location
        self.ims_location_min, self.ims_location_max = self.ims_data[:, :2].min(), self.ims_data[:, :2].max()
        self.dme_location_min, self.dme_location_max = self.dme_data[:, :2].min(), self.dme_data[:, :2].max()

        self.ims_data[:, :2] = (self.ims_data[:, :2] - self.ims_location_min) / (
                    self.ims_location_max - self.ims_location_min)
        self.dme_data[:, :2] = (self.dme_data[:, :2] - self.dme_location_min) / (
                self.dme_location_max - self.dme_location_min)

        # data
        self.ims_data_min, self.ims_data_max = self.ims_data[:, 2:].min(), self.ims_data[:, 2:].max()
        self.dme_data_min, self.dme_data_max = self.dme_data[:, 2:].min(), self.dme_data[:, 2:].max()

        self.ims_data[:, 2:] = (self.ims_data[:, 2:] - self.ims_data_min) / (
                self.ims_data_max - self.ims_data_min)
        self.dme_data[:, 2:] = (self.dme_data[:, 2:] - self.dme_data_min) / (
                self.dme_data_max - self.dme_data_min)

        # Transpose
        self.ims_data = self.ims_data.T
        self.dme_data = self.dme_data.T

        # set dim.
        self.m, self.k_m = self.dme_data.shape
        self.n, self.k_n = self.ims_data.shape

        print("Links dim: {} - has {} links, {} samples each, {} days \n"
              "Gueges dim: {} - has {} gauges, {} samples each, {} days"
            .format(
            self.dme_data.shape, self.k_m, self.m, config.coverage,
            self.ims_data.shape, self.k_n, self.n, config.coverage)
        )

    def tile_sum(self, tile, row, column, type):
        if np.isnan(tile).any():
            if type == 'dme':
                data_type = self.dme_data
                m = self.k_m
            elif type == 'ims':
                data_type = self.ims_data
                m = self.k_n
            else:
                raise ValueError('Unrecognized data type {}'.format(type))

            return np.nanmedian(data_type[row][np.clip(column - self.window, 0, m):np.clip(column + self.window, 0, m)])

        else:
            return np.sum(tile)

    def get_entry(self, arr, type):
        i = 0
        while (not ('name' in arr[i] and arr[i]['name'] == type)):
            i += 1

        return arr[i]

    def get_metadata(self, station_folder):
        f = open(config.ims_root_files + '/' + station_folder + '/' + 'metadata.txt', "r")
        metadata = {}
        try:
            for line in f:
                if 'location' in line:
                    location = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    metadata['latitude'] = location[0]
                    metadata['longitude'] = location[1]
        except IndexError:
            print("problem with metadata gague {}".format(station_folder))
            return

        return np.array([metadata['latitude'], metadata['longitude']])

    def load_ims_data(self):
        if not config.ims_pre_load_data:

            # 10[min] x 6 is an hour
            ims_matrix = np.empty((1, 6 * 24 * config.coverage + len(config.ims_metadata)))
            ims_order = np.empty(1)
            for index, station_folder in enumerate(os.listdir(config.ims_root_files)):
                print("now processing gauge: {}".format(station_folder))
                try:
                    df = pd.read_csv(config.ims_root_files + '/' + station_folder + '/' + 'data.csv')
                    ims_vec = self.get_metadata(station_folder)
                    if (ims_vec, df) is not (None, None):
                        for row in list(df.channels):
                            ims_vec = np.append(ims_vec,
                                                np.array([self.get_entry(ast.literal_eval(row), type='Rain')['value']]))

                        ims_vec = ims_vec.T
                        try:
                            ims_matrix = np.vstack((ims_matrix, ims_vec))
                            ims_order = np.hstack((ims_order, station_folder))
                        except ValueError:
                            print("problem with stacking gague {}".format(station_folder))

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_folder))

            ims_matrix = ims_matrix[1:]
            ims_order = ims_order[1:]

            if not os.path.isdir(config.ims_root_values + '/' + config.date_str_rep):
                os.makedirs(config.ims_root_values + '/' + config.date_str_rep)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'wb') as f:
                pickle.dump(ims_matrix, f)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'order.pkl', 'wb') as f:
                pickle.dump(ims_order, f)

        else:
            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'rb') as f:
                ims_matrix = pickle.load(f)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'order.pkl', 'rb') as f:
                ims_order = pickle.load(f)

        return ims_matrix, ims_order

    def load_dme_data(self):
        if not config.dme_pre_load_data:

            # 15[min] x 4 is an hour
            valid_row_number = 4 * 24 * config.coverage
            dme_matrix = np.empty((1, valid_row_number))
            dme_order = np.empty(1)

            for link in os.listdir(config.dme_root_files):

                link_name = link.split('_')[0]
                link_type = config.dme_scrape_config['link_objects']['measurement_type']

                print("preprocessing: now processing link: {} of type: {}".format(link_name, link_type))

                df = pd.read_csv(config.dme_root_files + '/' + link)

                df = df[df.Interval == 15]

                # todo: 'RFInputPower' is only good for one type of link
                if len(list(df['RFInputPower'])) > valid_row_number:
                    print(
                        'The provided data for link {} contains more rows then it should {}/{}'.format(link_name,
                                                                                                       len(list(
                                                                                                           df[
                                                                                                               'RFInputPower'])),
                                                                                                       valid_row_number))

                elif len(list(df['RFInputPower'])) == valid_row_number:
                    dme_matrix = np.vstack((dme_matrix, list(df['RFInputPower'])))
                    dme_order = np.hstack((dme_order, link_name))

            dme_matrix = dme_matrix[1:]
            dme_order = dme_order[1:]

            if not os.path.isdir(config.dme_root_values + '/' + config.date_str_rep):
                os.makedirs(config.dme_root_values + '/' + config.date_str_rep)

            with open(config.dme_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'wb') as f:
                pickle.dump(dme_matrix, f)
            with open(config.dme_root_values + '/' + config.date_str_rep + '/' + 'order.pkl', 'wb') as f:
                pickle.dump(dme_order, f)

        else:
            with open(config.dme_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'rb') as f:
                dme_matrix = pickle.load(f)
            with open(config.dme_root_values + '/' + config.date_str_rep + '/' + 'order.pkl', 'rb') as f:
                dme_order = pickle.load(f)

        return dme_matrix, dme_order


if __name__ == "__main__":
    Extractor()
