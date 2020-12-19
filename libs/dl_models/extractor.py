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
        self.ims_data,self.ims_order = self.load_ims_data()
        self.dme_data,self.dme_order = self.load_dme_data()

        #set the widndow for which the median will be calculated
        self.window=200

        # norm. - https://datascience.stackexchange.com/questions/5885/how-to-scale-an-array-of-signed-integers-to-range-from-0-to-1
        self.ims_data = (self.ims_data - self.ims_data.min()) / (self.ims_data.max() - self.ims_data.min())
        self.dme_data = (self.dme_data - np.nanmin(self.dme_data)) / (np.nanmax(self.dme_data) - np.nanmin(self.dme_data))

        # set dim.
        self.m, self.k_m = self.dme_data.shape
        self.n, self.k_n = self.ims_data.shape


        print("Before tile:\nX dim: {} - has {} links, {} samples each, {} days \n"
              "Y dim: {} - has {} gauges, {} samples each, {} days"
            .format(
            self.dme_data.shape, self.m, self.k_m, config.coverage,
            self.ims_data.shape, self.n, self.k_n, config.coverage)
        )

        print("\n----------\n")
        self.dme_data_tiled=np.array([ self.tile_sum(tile,row=row,column=column,type='dme') for column,batch in enumerate(np.hsplit(self.dme_data,self.k_m/2)) for row,tile in enumerate(batch)]).reshape(self.m, self.k_m//2)
        self.ims_data_tiled = np.nansum(np.hsplit(self.ims_data,self.k_n/3),axis=2).T

        print("After tile:\nX dim: {} - has {} links, {} samples each, {} days \n"
              "Y dim: {} - has {} gauges, {} samples each, {} days"
            .format(
            self.dme_data_tiled.shape, self.m, self.k_m/2, config.coverage,
            self.ims_ims_tiled.shape, self.n, self.k_n/3, config.coverage)
        )

    def tile_sum(self,tile,row,column,type):
        if np.isnan(tile).any():
            if type=='dme':
                data_type=self.dme_data
                m=self.k_m
            elif type=='ims':
                data_type=self.ims_data
                m=self.k_n
            else:
                raise ValueError('Unrecognized data type {}'.format(type))

            return np.nanmedian(data_type[row][np.clip(column - self.window, 0, m):np.clip(column + self.window, 0, m)])

        else:
            return np.sum(tile)

    def get_entry(self, arr):
        i = 0
        while (not ('name' in arr[i] and arr[i]['name'] == 'Rain')):
            i += 1

        return arr[i]

    def load_ims_data(self):
        if not config.ims_pre_load_data:

            # 10[min] x 6 is an hour
            ims_matrix = np.empty((1, 6 * 24 * config.coverage))
            ims_order=np.empty(1)
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
                        ims_order=np.hstack((ims_order,station_folder))
                    except ValueError:
                        print("problem with stacking gague {}".format(station_folder))

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_folder))

            ims_matrix = ims_matrix[1:]
            ims_order=ims_order[1:]

            if not os.path.isdir(config.ims_root_values + '/' + config.date_str_rep):
                os.makedirs(config.ims_root_values + '/' + config.date_str_rep)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'wb') as f:
                pickle.dump(ims_matrix, f)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'order.pkl', 'wb') as f:
                pickle.dump(ims_order,f)

        else:
            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'values.pkl', 'rb') as f:
                ims_matrix = pickle.load(f)

            with open(config.ims_root_values + '/' + config.date_str_rep + '/' + 'order.pkl', 'rb') as f:
                ims_order = pickle.load(f)

        return ims_matrix,ims_order

    def load_dme_data(self):
        if not config.dme_pre_load_data:

            # 15[min] x 4 is an hour
            valid_row_number = 4 * 24 * config.coverage
            dme_matrix = np.empty((1, valid_row_number))
            dme_order=np.empty(1)

            for link in os.listdir(config.dme_root_files):

                link_name = link.split('_')[0]
                link_type = config.dme_scrape_config['link_objects']['measurement_type']

                print("extractor: now processing link: {} of type: {}".format(link_name, link_type))

                df = pd.read_csv(config.dme_root_files + '/' + link)

                df = df[df.Interval == 15]
                time_value = config.date_datetime_rep

                if len(list(df['rain'])) > valid_row_number:
                    print(
                        'The provided data for link {} contains more rows then it should {}/{}'.format(link_name,
                            len(list(df['rain'])),valid_row_number))

                elif len(list(df['rain'])) == valid_row_number:
                    dme_matrix = np.vstack((dme_matrix, list(df['rain'])))
                    dme_order = np.hstack((dme_order, link_name))

                else:  # fill with nan for missing data

                    dme_vec = np.empty(1)
                    skip=False
                    for row_value, row_time, row_interval in zip(list(df['rain']), list(df.Time), list(df.Interval)):

                        # fill with rain data
                        if dme_vec[1:].size < valid_row_number:

                            if time_value > dt.strptime(row_time, "%Y-%m-%d %H:%M:%S"):
                                print('Something went wrong, extractor time missmatch, skipping {}...'.format(link_name))
                                skip=True
                                break


                            # fill with nan
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

                    while dme_vec[1:].size < valid_row_number and not skip:
                        dme_vec = np.hstack((dme_vec, np.nan))

                    if not skip:
                        dme_matrix = np.vstack((dme_matrix, dme_vec[1:].T))
                        dme_order=np.hstack((dme_order,link_name))

            dme_matrix = dme_matrix[1:]
            dme_order=dme_order[1:]

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

        return dme_matrix,dme_order


if __name__ == "__main__":
    Extractor()
