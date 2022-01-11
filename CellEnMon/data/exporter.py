import CellEnMon.config as config
import pickle
import numpy as np
import pandas as pd
import os
import ast


class Domain:
    def __init__(self, db, metadata_vector_len):
        self.db = db
        self.metadat_vector_len = metadata_vector_len
        self.metadata, self.data = db[:metadata_vector_len], db[metadata_vector_len:]
        self.data_max, self.data_min, self.data_normalized = self.normalizer(self.data)
        self.metadata_max, self.metadata_min, metadata_normalized = self.normalizer(self.metadata)

    def normalizer(self, mat):
        min = mat.min()
        max = mat.max()
        mat = 0 if max - min == 0 else (mat - min) / (max - min)
        return min, max, mat


class Extractor:
    def __init__(self):
        self.dme = Domain(*self.load_dme())  # 1 day is 96 = 24*4 data samples + 7 metadata samples
        self.ims = Domain(*self.load_ims())  # 1 day is 144 = 24*6 data samples + 2 metadata samples

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
        station_name_splited = station_name.split()
        metadata["latitude"] = station_name_splited[3]
        metadata["longitude"] = station_name_splited[4]
        metadata["vector"] = np.array([metadata['latitude'], metadata['longitude']])
        return metadata

    def load_ims(self):
        temp_str = f'{config.ims_root_files}/processed/{config.date_str_rep}'

        try:
            with open(f'{temp_str}/values.pkl', 'rb') as f:
                ims_matrix = pickle.load(f)

        except FileNotFoundError:

            # 10[min] x 6 is an hour
            ims_matrix = np.empty((1, 6 * 24 * config.coverage + len(config.ims_metadata)))
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
                        except ValueError:
                            print("problem with stacking gague {}".format(station_file_name))

                except FileNotFoundError:
                    print("data does not exist in {}".format(station_file_name))

            ims_matrix = ims_matrix[1:]

            if not os.path.isdir(temp_str):
                os.makedirs(temp_str)

            with open(f'{temp_str}/values.pkl', 'wb') as f:
                pickle.dump(ims_matrix, f)

        return ims_matrix.astype(np.float), len(metadata["vector"])

    def get_dme_metadata(self, link_file_name):
        metadata = {}
        link_file_name_splited = link_file_name.split()
        metadata["source"] = f'{link_file_name_splited[0]}'
        metadata["sink"] = f'{link_file_name_splited[3]}'
        metadata["link_name"] = f'{metadata["source"]}-{metadata["sink"]}'
        metadata["tx_longitude"] = link_file_name_splited[1]
        metadata["tx_latitude"] = link_file_name_splited[2]
        metadata["rx_longitude"] = link_file_name_splited[4]
        metadata["rx_latitude"] = link_file_name_splited[5]
        metadata["vector"] = [metadata["tx_longitude"], metadata["tx_latitude"], metadata["rx_longitude"],
                              metadata["rx_latitude"]]

        return metadata

    def load_dme(self):
        temp_str = f'{config.dme_root_files}/processed/{config.date_str_rep}'
        try:
            with open(f'{temp_str}/values.pkl', 'rb') as f:
                dme_matrix = pickle.load(f)

        except FileNotFoundError:

            # 15[min] x 4 is an hour
            valid_row_number = 4 * 24 * config.coverage
            dme_matrix = np.empty((1, valid_row_number + len(config.dme_metadata)))

            for link_file_name in os.listdir(config.dme_root_files):

                link_metadata = self.get_dme_metadata(link_file_name)
                link_type = config.dme_scrape_config['link_objects']['measurement_type']

                print(
                    "preprocessing: now processing link: {} of type: {}".format(link_metadata["link_name"], link_type))

                df = pd.read_csv(f"{config.dme_root_files}/raw/{link_file_name}")
                df = df[df.Interval == 15]

                # todo: 'RFInputPower' is only good for one type of link
                if len(list(df['RFInputPower'])) > valid_row_number:
                    print(
                        'The provided data for link {} contains more rows then it should {}/{}'.format(link_metadata["link_name"],
                                                                                                       len(list(
                                                                                                           df[
                                                                                                               'RFInputPower'])),
                                                                                                       valid_row_number))

                elif len(list(df['RFInputPower'])) == valid_row_number:
                    dme_matrix = np.vstack(
                        (dme_matrix, link_metadata["vector"] + list(df['RFInputPower'])))

            dme_matrix = dme_matrix[1:]

            if not os.path.isdir(temp_str):
                os.makedirs(temp_str)

            with open(f'{temp_str}/values.pkl', 'wb') as f:
                pickle.dump(dme_matrix, f)

        return dme_matrix.astype(np.float), len(link_metadata["vector"])


if __name__ == "__main__":
    dataset = Extractor()
    print(dataset.stats())
