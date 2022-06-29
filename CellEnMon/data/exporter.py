import math

import CellEnMon.config as config
import pickle
import numpy as np
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import time
from sklearn.model_selection import train_test_split


class Domain:
    def __init__(self, db, db_type):
        self.station_names_vec = db.keys()
        self.db = db
        self.db_normalized = {}

        for station_name, value in db.items():
            data_max, data_min, data_normalized = self.normalizer(np.array(list(value['data'].values())))
            metadata_max, metadata_min, metadata_normalized = self.normalizer(value['metadata'])
            self.db_normalized[station_name] = {
                "data": dict(zip(np.array(list(value['data'].keys())), data_normalized)),
                "time": np.array(list(value['data'].keys())),
                "data_min": data_min,
                "data_max": data_max,
                "metadata": metadata_normalized,
                "metadata_max": metadata_max,
                "metadata_min": metadata_min,
            }
        self.df = pd.DataFrame.from_dict(self.db_normalized)

    def normalizer(self, mat):
        min = mat.min()
        max = mat.max()
        mat = 0 if max - min == 0 else (mat - min) / (max - min)
        return min, max, mat


class Extractor:
    def __init__(self,is_train=True):
        self.dme = Domain(self.load_dme(is_train=is_train), db_type="dme")  # 1 day is 96 = 24*4 data samples + 7 metadata samples
        self.ims = Domain(self.load_ims(is_train=is_train), db_type="ims")  # 1 day is 144 = 24*6 data samples + 2 metadata samples

        # a * np.exp(-b * x) + c
        self.a = None
        self.b = None
        self.c = None

    def visualize_ims(self, gauge_name=None):
        if gauge_name in self.ims.db:
            x = list(self.ims.db[gauge_name]['data'].keys())
            RR = list(self.ims.db[gauge_name]['data'].values())
            plt.plot(x, RR)

            plt.xticks(x[::250], rotation=45)
            plt.legend(["RR"])
            plt.title("Rain Rate")
            plt.xlabel("TimeStamp")
            plt.ylabel("mm/h")
            plt.show()
        else:
            raise FileNotFoundError(f"The provided gague_name:{gauge_name} doesn't exist in the ims dataset")

    def extract_TSL_RSL(self, data):
        return [x[0] for x in data], [x[1] for x in data], [x[2] for x in data], [x[3] for x in data]

    def visualize_dme(self, link_name=None):
        if link_name in self.dme.db:
            x = list(self.dme.db[link_name]['data'].keys())
            MTSL, mTSL, MRSL, mRSL = self.extract_TSL_RSL(list(self.dme.db[link_name]['data'].values()))
            plt.plot(x, MRSL)
            plt.plot(x, mRSL)

            plt.xticks(x[::100], rotation=45)
            plt.legend(["MRSL", "mRSL"])
            plt.title("Receive Signal Level")
            plt.xlabel("TimeStamp")
            plt.ylabel("dBm")
            plt.show()
        else:
            raise FileNotFoundError(f"The provided link_name:{link_name} doesn't exist in the dme dataset")

    def calculate_wet_events_histogram(self):
        return np.array([y for x in self.ims.db for y in list(self.ims.db[x]['data'].values())])

    def func_fit(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def stats(self):
        wet_events_hist = self.calculate_wet_events_histogram()
        wet_events_precentage = len(wet_events_hist[wet_events_hist > 0]) / len(wet_events_hist)

        counts, bins = np.histogram(wet_events_hist)
        counts = [x / sum(counts) for x in counts]
        plt.hist(bins[:-1], bins, weights=counts, rwidth=0.7, label="Rain Rate Histogram")

        plt.title("Rain Rate All of Israel")
        plt.xlabel("Rain Rate [mm/h]")
        plt.ylabel("$f_{RR}(r)$")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        # Plotting exp. fit
        popt, pcov = curve_fit(self.func_fit, bins[:-1], counts / sum(counts))
        self.a, self.b, self.c = popt
        plt.plot(bins, self.func_fit(bins, *popt), 'r-',
                 label='fit[a * exp(-b * x) + c]: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.yscale("log")
        plt.legend()

        print(
            f"start:{config.start_date_str_rep_ddmmyyyy} end:{config.end_date_str_rep_ddmmyyyy} --- in total it is {config.coverage} days\n" \
            f"🖇️ Links: {len(self.dme.station_names_vec)} 🖇️\n" \
            f"🏺 Gauges: {len(self.ims.station_names_vec)} 🏺\n" \
            f"\U0001F4A6 Wet:{wet_events_precentage} \U0001F4A6 \n" \
            f"\U0001F975 Dry:{1 - wet_events_precentage} \U0001F975 \n\n" \
            f"\U0001F9EA Exp fit:{popt} \U0001F9EA")

    def get_entry(self, arr, type):
        i = 0
        while (not ('name' in arr[i] and arr[i]['name'] == type)):
            i += 1

        return arr[i]

    def get_ims_metadata(self, station_name):
        metadata = {}
        station_name_splited = station_name.split('-')
        metadata["logitude"] = station_name_splited[3]
        metadata["latitude"] = station_name_splited[4].replace(".csv", "")
        metadata["gauge_name"] = f"{station_name_splited[0]}-{station_name_splited[1]}-{station_name_splited[2]}"
        metadata["vector"] = np.array([float(metadata['logitude']), float(metadata['latitude'])])
        return metadata

    def load_ims(self, is_train=False):
        dataset_type_str = "train" if is_train else "validation"
        temp_str = f'{config.ims_root_files}/processed'
        try:
            with open(f'{temp_str}/{dataset_type_str}.pkl', 'rb') as f:
                dataset = pickle.load(f)

            if not os.path.isdir(temp_str):
                os.makedirs(temp_str)

        except FileNotFoundError:
            # 10[min] x 6 is an hour
            ims_matrix = {}
            for index, station_file_name in enumerate(os.listdir(f'{config.ims_root_files}/raw')):
                print("now processing gauge: {}".format(station_file_name))
                try:
                    df = pd.read_csv(f'{config.ims_root_files}/raw/{station_file_name}')
                    metadata = self.get_ims_metadata(f'{station_file_name}')
                    ims_vec = np.array([])
                    time = np.array([" ".join(t.split('+')[0].split('T')) for t in df.datetime])

                    if (ims_vec, df) is not (None, None):
                        for row in list(df.channels):
                            ims_vec = np.append(ims_vec,
                                                np.array([self.get_entry(ast.literal_eval(row), type='Rain')['value']]))

                        ims_matrix[metadata["gauge_name"]] = \
                            {
                                "metadata_len": len(metadata["vector"]),
                                "data_len": len(ims_vec),
                                "data": dict(zip(time, ims_vec)),
                                "metadata": metadata["vector"]
                            }

                        data = {'Time': time, 'RainAmout[mm/h]': ims_vec}
                        pd.DataFrame.from_dict(data).to_csv(f"{temp_str}/{station_file_name}", index=False)

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_file_name))

            s = pd.Series(ims_matrix)
            training_data, validation_data = [i.to_dict() for i in train_test_split(s, train_size=0.7)]
            dataset = training_data if is_train else validation_data
            with open(f'{temp_str}/{dataset_type_str}.pkl', 'wb') as f:
                pickle.dump(dataset, f)

        return dataset

    def get_dme_metadata(self, link_file_name):
        metadata = {}
        link_file_name_splited = link_file_name.split('-')
        metadata["source"] = f'{link_file_name_splited[0]}'
        metadata["sink"] = f'{link_file_name_splited[3]}'
        metadata["link_name"] = f'{metadata["source"]}-{metadata["sink"]}'
        metadata["tx_longitude"] = link_file_name_splited[1]
        metadata["tx_latitude"] = link_file_name_splited[2]
        metadata["rx_longitude"] = link_file_name_splited[4]
        metadata["rx_latitude"] = link_file_name_splited[5].replace(".csv", "")
        metadata["vector"] = np.array([float(metadata["tx_longitude"]),
                                       float(metadata["tx_latitude"]),
                                       float(metadata["rx_longitude"]),
                                       float(metadata["rx_latitude"])])

        return metadata

    def load_dme(self, is_train=False):
        dataset_type_str = "train" if is_train else "validation"
        temp_str = f'{config.dme_root_files}/processed'
        try:
            with open(f'{temp_str}/{dataset_type_str}.pkl', 'rb') as f:
                dataset = pickle.load(f)

            if not os.path.isdir(temp_str):
                os.makedirs(temp_str)

        except FileNotFoundError:

            # 15[min] x 4 is an hour
            valid_row_number = 4 * 24 * config.coverage
            dme_matrix = {}

            for link_file_name in os.listdir(f'{config.dme_root_files}/raw'):

                metadata = self.get_dme_metadata(link_file_name)
                print(f"preprocessing: now processing link: {metadata['link_name']}")

                df = pd.read_csv(f"{config.dme_root_files}/raw/{link_file_name}")
                df = df[df.Interval == 15]

                if len(list(df['Time'])) < valid_row_number:
                    print(
                        f'Number of rows for link: {metadata["link_name"]} is wrong: {len(list(df["Time"]))}<{valid_row_number}')

                else:
                    # if 'RFInputPower' in df:
                    #     stack = [link_metadata["vector"] + x for x in [df.RFInputPower, df.RFOutputPower]]
                    if 'PowerTLTMmax' in df and 'PowerTLTMmin' in df and 'PowerRLTMmax' in df and 'PowerRLTMmax' in df:
                        try:
                            PowerTLTMmax = np.array(df[~df.PowerTLTMmax.isnull()].PowerTLTMmax.astype(int))
                            PowerTLTMmin = np.array(df[~df.PowerTLTMmin.isnull()].PowerTLTMmin.astype(int))
                            PowerRLTMmax = np.array(df[~df.PowerRLTMmax.isnull()].PowerRLTMmax.astype(int))
                            PowerRLTMmin = np.array(df[~df.PowerRLTMmin.isnull()].PowerRLTMmin.astype(int))
                            Time = np.array(df[~df.PowerRLTMmin.isnull()].Time)
                            data = np.vstack((PowerTLTMmax, PowerTLTMmin, PowerRLTMmax, PowerRLTMmin)).T

                            dme_matrix[metadata["link_name"]] = {
                                'metadata_len': len(metadata["vector"]),
                                'data_len': len(PowerRLTMmin),
                                "data": dict(zip(Time, data)),
                                "metadata": metadata["vector"]
                            }

                            data = {'Time': Time, 'PowerTLTMmax[dBm]': PowerTLTMmax, 'PowerTLTMmin[dBm]': PowerTLTMmin,
                                    'PowerRLTMmax[dBm]': PowerRLTMmax, 'PowerRLTMmin[dBm]': PowerRLTMmin}
                            pd.DataFrame.from_dict(data).to_csv(f"{temp_str}/{link_file_name}", index=False)

                        except ValueError:
                            print(
                                f"link's:{metadata['link_name']} dim are not compatible: PowerTLTMmax:{len(PowerTLTMmax)} | PowerTLTMmin:{len(PowerTLTMmin)} | PowerRLTMmax:{len(PowerRLTMmax)} | PowerRLTMmin:{len(PowerRLTMmin)}")

                    else:
                        print(
                            f"Not all fields [PowerTLTMmax | PowerTLTMmin | PowerRLTMmax | PowerRLTMmax] were provided in link:{metadata['link_name']}")

            s = pd.Series(dme_matrix)
            training_data, validation_data = [i.to_dict() for i in train_test_split(s, train_size=0.7)]
            dataset = training_data if is_train else validation_data
            with open(f'{temp_str}/{dataset_type_str}.pkl', 'wb') as f:
                pickle.dump(dataset, f)


        return dataset


if __name__ == "__main__":
    dataset = Extractor()
    dataset.stats()
    # dataset.visualize_dme(link_name='a459-6879')
    # dataset.visualize_ims(gauge_name='71-232-NEOT SMADAR')
    plt.show()
