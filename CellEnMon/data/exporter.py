import CellEnMon.config as config
import pickle
import numpy as np
import pandas as pd
import os
import ast


class Domain:
    def __init__(self, db):
        self.station_names_vec = db.keys()
        self.db = db
        self.db_normalized={}
        for station_name,value in db.items():
            data_max, data_min, data_normalized=self.normalizer(value['data'])
            metadata_max, metadata_min, metadata_normalized = self.normalizer(value['metadata'])
            self.db_normalized[station_name]={
                "data": data_normalized,
                "time": value['time'],
                "data_min":data_min,
                "data_max":data_max,
                "metadata":  metadata_normalized,
                "metadata_max": metadata_max,
                "metadata_min": metadata_min,
            }

    def normalizer(self, mat):
        min = mat.min()
        max = mat.max()
        mat = 0 if max - min == 0 else (mat - min) / (max - min)
        return min, max, mat


class Extractor:
    def __init__(self):
        self.dme = Domain(self.load_dme())  # 1 day is 96 = 24*4 data samples + 7 metadata samples
        self.ims = Domain(self.load_ims())  # 1 day is 144 = 24*6 data samples + 2 metadata samples

    def stats(self):
        message = f"start:{config.start_date_str_rep} end:{config.end_date_str_rep} --- in total it is {config.coverage} days" \
                  "🚀 Links 🚀" \
                  f"Link matrix shape: {self.dme.db.shape}" \
                  f"This means that we have: {self.dme.db.shape[0]} links; each link is a vector of len:{self.dme.data.shape[1]}" \
                  f"A link's vector is composed of #metadata:{self.dme.metadat_vector_len} and #data:{self.dme.db.shape[0] - self.dme.metadat_vector_len}" \
                  "\n\n" \
                  "🏺 Gauges 🏺" \
                  f"Guege matrix shape: {self.ims.data.shape}" \
                  f"This means that we have: {self.ims.data.shape[0]} gauges; each gauges is a vector of len:{self.ims.data.shape[1]}" \
                  f"A gauge is a vector composed of metadata:{self.ims.metadat_vector_len} and data:{self.ims.db.shape[0] - self.ims.metadat_vector_len}"

        return message

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
        metadata["gague_name"] = f"{station_name_splited[0]}-{station_name_splited[1]}-{station_name_splited[2]}"
        metadata["vector"] = np.array([float(metadata['longitude']), float(metadata['latitude'])])
        return metadata

    def load_ims(self):
        temp_str = f'{config.ims_root_files}/processed'

        try:
            with open(f'{temp_str}/values.pkl', 'rb') as f:
                ims_matrix = pickle.load(f)

        except FileNotFoundError:

            # 10[min] x 6 is an hour
            ims_matrix = {}
            for index, station_file_name in enumerate(os.listdir(f'{config.ims_root_files}/raw')):
                print("now processing gauge: {}".format(station_file_name))
                try:
                    df = pd.read_csv(station_file_name)
                    metadata = self.get_ims_metadata(station_file_name)
                    ims_vec = metadata["vector"]

                    if (ims_vec, df) is not (None, None):
                        for row in list(df.channels):
                            ims_vec = np.append(ims_vec,
                                                np.array([self.get_entry(ast.literal_eval(row), type='Rain')['value']]))

                        ims_vec = ims_vec.T
                        try:
                            ims_matrix = np.vstack((ims_matrix, ims_vec))
                            ims_matrix[metadata["gauge_name"]] = \
                                {
                                    "metadata_len": len(metadata["vector"]),
                                    "data_len": len(ims_vec),
                                    "data": ims_vec,
                                    "time": Time,
                                    "metadata": metadata["vector"]
                                }

                        except ValueError:
                            print("problem with stacking gague {}".format(station_file_name))

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_file_name))

            if not os.path.isdir(temp_str):
                os.makedirs(temp_str)

            with open(f'{temp_str}/values.pkl', 'wb') as f:
                pickle.dump(ims_matrix, f)

        return ims_matrix

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
        metadata["vector"] = [float(metadata["tx_longitude"]), float(metadata["tx_latitude"]),
                              float(metadata["rx_longitude"]),
                              float(metadata["rx_latitude"])]

        return metadata

    def load_dme(self):
        temp_str = f'{config.dme_root_files}/processed'
        try:
            with open(f'{temp_str}/values.pkl', 'rb') as f:
                dme_matrix = pickle.load(f)

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
                                'data_len' : len(PowerRLTMmin),
                                "data": data,
                                "time": Time,
                                "metadata": metadata["vector"]
                            }

                        except ValueError:
                            print(
                                f"link's:{metadata['link_name']} dim are not compatible: PowerTLTMmax:{len(PowerTLTMmax)} | PowerTLTMmin:{len(PowerTLTMmin)} | PowerRLTMmax:{len(PowerRLTMmax)} | PowerRLTMmin:{len(PowerRLTMmin)}")

                    else:
                        print(
                            f"Not all fields [PowerTLTMmax | PowerTLTMmin | PowerRLTMmax | PowerRLTMmax] were provided in link:{metadata['link_name']}")

            if not os.path.isdir(temp_str):
                os.makedirs(temp_str)

            with open(f'{temp_str}/values.pkl', 'wb') as f:
                pickle.dump(dme_matrix, f)

        return dme_matrix


if __name__ == "__main__":
    dataset = Extractor()
    print(dataset.stats())
