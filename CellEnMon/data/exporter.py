import math
import sys

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

IS_TRAIN = False
CROSS_DOMAIN_METADATA_NORMALIZATION=True


class Domain:
    def __init__(self, db, db_type):
        self.station_names_vec = db.keys()
        self.db = db
        self.db_normalized = {}
        self.db_type = db_type
        self.metadata_long_max = -sys.maxsize
        self.metadata_long_min = sys.maxsize
        self.metadata_lat_max = -sys.maxsize
        self.metadata_lat_min = sys.maxsize
        self.data_min=999
        self.data_max=-999

        # Data Normalization
        for station_name, value in db.items():
            data_max, data_min, data_normalized = self.normalizer(np.array(list(value['data'].values())))
            self.db_normalized[station_name] = {
                "data": dict(zip(np.array(list(value['data'].keys())), data_normalized)),
                "time": np.array(list(value['data'].keys())),
                "data_min": data_min,
                "data_max": data_max,
            }
            # Find min-max for metadata normalization
            self.metadata_min_max_finder(value['metadata'])
            self.data_min=min(self.data_min, data_min)
            self.data_max=max(self.data_max, data_max)

        self.df = pd.DataFrame.from_dict(self.db_normalized)
        print(f"min-max:{self.data_min},{self.data_max}")

    def metadata_normalization(self):
        for station_name, value in self.db.items():
            self.db_normalized[station_name]["norm_metadata"] = value['metadata'] #self.min_max_norm(

    def metadata_min_max_finder(self, metadata_vector):
        self.metadata_long_max = max(self.metadata_long_max, metadata_vector[0], metadata_vector[2])
        self.metadata_long_min = min(self.metadata_long_min, metadata_vector[0], metadata_vector[2])
        self.metadata_lat_max = max(self.metadata_lat_max, metadata_vector[1], metadata_vector[3])
        self.metadata_lat_min = min(self.metadata_lat_min, metadata_vector[1], metadata_vector[3])

    def min_max_norm(self, x):
        y=[0,0,0,0]
        y[0] = self.norm(x[0], self.metadata_long_min, self.metadata_long_max)
        y[1] = self.norm(x[1], self.metadata_lat_min, self.metadata_lat_max)
        y[2] = self.norm(x[2], self.metadata_long_min, self.metadata_long_max)
        y[3] = self.norm(x[3], self.metadata_lat_min, self.metadata_lat_max)
        return y

    def norm(self, x, mmin, mmax):
        epsilon=1e-6
        return ((x - mmin) / (mmax - mmin + epsilon))

    def normalizer(self, mat):
        min = -50.8 if self.db_type=="dme" else 0
        max = 17 if self.db_type=="dme" else 3.3
        mat=mat #(mat - min) / (max - min)

        return max, min, mat


class Extractor:
    def __init__(self, is_train=True):
        self.export_type = config.export_type
        self.dme = Domain(self.load_dme(is_train=is_train),
                          db_type="dme")  # 1 day is 96 = 24*4 data samples + 7 metadata samples
        self.ims = Domain(self.load_ims(is_train=is_train),
                          db_type="ims")  # 1 day is 144 = 24*6 data samples + 2 metadata samples

        self.metadata_long_max = max(self.dme.metadata_long_max,self.ims.metadata_long_max)
        self.metadata_long_min = min(self.dme.metadata_long_min,self.ims.metadata_long_min)
        self.metadata_lat_max = max(self.dme.metadata_lat_max,self.ims.metadata_lat_max)
        self.metadata_lat_min = min(self.dme.metadata_lat_min,self.ims.metadata_lat_min)
        

        if CROSS_DOMAIN_METADATA_NORMALIZATION:
            #dme
            self.dme.metadata_long_max=self.metadata_long_max
            self.dme.metadata_long_min=self.metadata_long_min
            self.dme.metadata_lat_max=self.metadata_lat_max
            self.dme.metadata_lat_min = self.metadata_lat_min

            #ims
            self.ims.metadata_long_max = self.metadata_long_max
            self.ims.metadata_long_min = self.metadata_long_min
            self.ims.metadata_lat_max = self.metadata_lat_max
            self.ims.metadata_lat_min = self.metadata_lat_min

        self.dme.metadata_normalization()
        self.ims.metadata_normalization()

        # a * np.exp(-b * x) + c
        self.rain_a = None
        self.rain_b = None
        self.rain_c = None
        self.attenuation_a = None
        self.attenuation_b = None
        self.attenuation_c = None

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
    
    def calculate_attenuation_events_histogram(self):
        return np.array([y for x in self.dme.db for y in list(self.dme.db[x]['data'].values())])[:,-1]

    def func_fit(self, x, a):
        return a * np.exp(-x * a)

    def stats(self):
        #rain
        wet_events_hist = self.calculate_wet_events_histogram()
        wet_events_precentage = len(wet_events_hist[wet_events_hist > 0.2]) / len(wet_events_hist)

        counts, bins = np.histogram(wet_events_hist)
        counts = [x / sum(counts) for x in counts]

        # plt.hist(bins[:-1], bins, weights=counts, rwidth=0.7, label="Rain Rate Histogram")
        # plt.title("Rain Rate All of Israel")
        # plt.xlabel("Rain Rate [mm/h]")
        # plt.ylabel("$f_{RR}(r)$")
        # plt.grid(color='gray', linestyle='-', linewidth=0.5)

        # Plotting exp. fit
        popt, pcov = curve_fit(self.func_fit, bins[:-1], counts / sum(counts))
        self.rain_a = popt
        # plt.plot(bins, self.func_fit(bins, *popt), 'r-',
        #          label='fit[a * exp(-a * x) ]: a=%5.3f' % tuple(popt))
        # plt.yscale("log")
        # plt.legend()
        
        
        #attenuation
        attenuation_events_hist=self.calculate_attenuation_events_histogram()
        attenuation_events_precentage=len(attenuation_events_hist[attenuation_events_hist>0])/len(attenuation_events_hist)
        counts, bins = np.histogram(attenuation_events_hist)
        counts = [x / sum(counts) for x in counts]
        self.attenuation_a = popt
    
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
        try:
            station_name_splited = station_name.split('_')
            metadata["logitude"] = station_name_splited[1]
            metadata["latitude"] = station_name_splited[2].replace(".csv", "")
            metadata["gauge_name"] = f"{station_name_splited[0]}"
            metadata["vector"] = np.array(
                [float(metadata['logitude']), float(metadata['latitude']), float(metadata['logitude']),
                 float(metadata['latitude'])])
        
        except IndexError:
            print(f"File is not in correct format:{station_name}")
            return 
        
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
                    metadata = self.get_ims_metadata(f'{station_file_name}')
                    if metadata:
                        df = pd.read_csv(f'{config.ims_root_files}/raw/{station_file_name}')
                        
                        time=df.Time.to_numpy()
                        ims_vec=df["RainAmout[mm/h]"].to_numpy()
                            
                        
                        ims_matrix[metadata["gauge_name"]] = \
                        {
                            "metadata_len": len(metadata["vector"]),
                            "data_len": len(ims_vec),
                            "data": dict(zip(time, ims_vec)),
                            "metadata": metadata["vector"]
                        }

                
                        data = {'Time': time, 'RR[mm/h]': ims_vec}
                        pd.DataFrame.from_dict(data).to_csv(f"{temp_str}/{station_file_name}", index=False)

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_file_name))
            
            s = pd.Series(ims_matrix)

            training_data, validation_data = [i.to_dict() for i in train_test_split(s, test_size=0.000001, shuffle=False)]
            
            #Conditional dataset
            validation_data["LAHAV"]=training_data["LAHAV"]
            validation_data["NEOT SMADAR"]=training_data["NEOT SMADAR"]
            
            #train pop
            training_data.pop("LAHAV",None)
            training_data.pop("NEOT SMADAR",None)

            #validation pop
            validation_data.pop("ZOMVET HANEGEV",None)
            
            dataset = training_data if is_train else validation_data
                            
            
            with open(f'{temp_str}/{dataset_type_str}.pkl', 'wb') as f:
                pickle.dump(dataset, f)

        return dataset

    def get_dme_metadata(self, link_file_name):
        metadata = {}
        link_file_name_splited = link_file_name.split('_')
        try:
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
        except IndexError:
            print(f"File is not in correct format:{link_file_name}")
            return

        return metadata
    
    def smoothing(self, arr, n):
        return np.nanmean(np.pad(arr.astype(float), (0, n - arr.size%n), mode='constant', constant_values=arr[0]).reshape(-1, n), axis=1)
    
    def load_dme(self, is_train=False):
        dataset_type_str = "train" if is_train else "validation"
        smoothing_n=config.smoothing_dme
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
                print(f"metadata:{metadata}")
                if metadata:
                    #print(f"preprocessing: now processing link: {metadata['link_name']}")

                    df = pd.read_csv(f"{config.dme_root_files}/raw/{link_file_name}")
                    #print(f"df:{df.head()}")
                    if 'PowerRLTMmax[dBm]_baseline' in df and 'PowerRLTMmin[dBm]_baseline' in df :
                        

                        PowerTLTMmax = np.array(df[~df["PowerTLTMmax[dBm]_baseline"].isnull()]["PowerTLTMmax[dBm]_baseline"].astype(float))

                        PowerTLTMmin = np.array(df[~df["PowerTLTMmin[dBm]_baseline"].isnull()]["PowerTLTMmin[dBm]_baseline"].astype(float))

                        PowerRLTMmax = np.array(df[~df["PowerRLTMmax[dBm]_baseline"].isnull()]["PowerRLTMmax[dBm]_baseline"].astype(float))

                        PowerRLTMmin = np.array(df[~df["PowerRLTMmin[dBm]_baseline"].isnull()]["PowerRLTMmin[dBm]_baseline"].astype(float))

                        
                        

                        Time = df.Time.to_numpy()
                        data = np.vstack((PowerTLTMmax, PowerTLTMmin, PowerRLTMmax, PowerRLTMmin)).T #
                          
                        
                        #print(f"data:{data}")
                        
                        if len(PowerRLTMmax)==len(PowerRLTMmin) and len(PowerRLTMmin)==len(Time):
                            #len(PowerTLTMmax)==len(PowerTLTMmin) and len(PowerTLTMmin)==len(PowerRLTMmax) and len(PowerRLTMmax)==len(PowerRLTMmin) and
                          
                            dme_matrix[metadata["link_name"]] = {
                                'metadata_len': len(metadata["vector"]),
                                'data_len': len(PowerRLTMmin),
                                "data": dict(zip(Time, data)),
                                "metadata": metadata["vector"]
                            }

                            data = {'Time': Time, 'PowerTLTMmax[dBm]': PowerTLTMmax, 'PowerTLTMmin[dBm]': PowerTLTMmin, 'PowerRLTMmax[dBm]': PowerRLTMmax, 'PowerRLTMmin[dBm]': PowerRLTMmin}
                            pd.DataFrame.from_dict(data).to_csv(f"{temp_str}/{link_file_name}", index=False)



                    else:
                        print(
                            f"Not all fields [PowerRLTMmax | PowerRLTMmax] were provided in link:{metadata['link_name']}")

            s = pd.Series(dme_matrix)
            training_data, validation_data = [i.to_dict() for i in train_test_split(s, test_size=0.000001, shuffle=False)]
            
            #Conditional dataset
            validation_data["b394-ts04"]=training_data["b394-ts04"]
            training_data.pop("b394-ts04",None)
            
            dataset = training_data if is_train else validation_data
            with open(f'{temp_str}/{dataset_type_str}.pkl', 'wb') as f:
                pickle.dump(dataset, f)

        return dataset


if __name__ == "__main__":
    dataset = Extractor(is_train=IS_TRAIN)
    #dataset.visualize_dme(link_name='a459-6879')
    #dataset.visualize_ims(gauge_name='71-232-NEOT SMADAR')

    dataset.stats()
    plt.show()
    
