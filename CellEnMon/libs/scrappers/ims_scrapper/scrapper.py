import json
import requests
import pandas as pd
import CellEnMon.config as config
import os
from datetime import datetime as dt
from google.cloud import storage

url = "https://api.ims.gov.il/v1/envista/stations/64/data?from=2019/12/01&to=2020/01/01"

headers = {
    'Authorization': 'ApiToken ' + config.ims_token
}


## Setting credentials using the downloaded JSON file
path = 'cellenmon-e840a9ba53e8.json'
if not os.path.isfile(path):
    raise ("Please provide the gcs key in the root directory")
client = storage.Client.from_service_account_json(json_credentials_path=path)

## For slow upload speed
storage.blob._DEFAULT_CHUNKSIZE = 2097152  # 1024 * 1024 B * 2 = 2 MB
storage.blob._MAX_MULTIPART_SIZE = 2097152  # 2 MB

SELECTOR=['UPLOAD'] #'DOWNLOAD', 'UPLOAD'

class IMS_Scrapper_obj:
    def __init__(self, index, station_id, _from, _to):
        self.index = index
        self.station_id = station_id
        self._from = _from
        self._to = _to
        self.root = config.ims_root_files
        self.station_meta_data = f"https://api.ims.gov.il/v1/envista/stations/{station_id}"
        self.station_data = f"https://api.ims.gov.il/v1/envista/stations/{station_id}/data/?from={_from}&to={_to}"
        self.bucket = client.get_bucket('cell_en_mon')

    def upload_files_to_gcs(self):
        root=f"datasets/ims/raw/{config.start_date_str_rep}_{config.end_date_str_rep}"
        for station in os.listdir(root):
            for file in os.listdir(f'{root}/{station}'):
                blob = self.bucket.blob(f'ims/{config.start_date_str_rep}-{config.end_date_str_rep}/raw/{station}/{file}')
                try:
                    with open(f"{root}/{station}/{file}", 'rb') as f:
                        blob.upload_from_file(f)
                    print(f'Uploaded file:{file} succesfully !')
                except Exception as e:
                    print(f'Uploaded file:{file} failed with the following exception:{e}!')

    def download_from_ims(self):
        metadata_response = requests.request("GET", self.station_meta_data, headers=headers)
        data_response = requests.request("GET", self.station_data, headers=headers)

        if metadata_response.status_code != 200 or data_response.status_code != 200:
            print("station id: {} , metadata respose: {} , data response: {}".format(self.station_id,
                                                                                     metadata_response.status_code,
                                                                                     data_response.status_code))

        else:
            metadata = json.loads(metadata_response.text.encode('utf8'))
            data = json.loads(data_response.text.encode('utf8'))

            folder = "{}-{}-{}".format(self.index, metadata['stationId'], metadata['name'])
            if not os.path.exists(self.root + '/' + folder):
                os.makedirs(self.root + '/' + folder)
            else:
                try:
                    os.remove(self.root + '/' + 'monitors.csv')
                    os.remove(self.root + '/' + 'metadata.txt')
                    os.remove(self.root + '/' + 'data.csv')
                except FileNotFoundError:
                    pass

            pd.DataFrame(metadata['monitors']).to_csv(self.root + '/' + folder + '/' + "monitors.csv", index=False)
            pd.DataFrame(data['data']).to_csv(self.root + '/' + folder + '/' + "data.csv", index=False)
            with open(self.root + '/' + folder + '/' + "metadata.txt", 'w') as file:
                file.write('stationId: {}\n'.format(metadata['stationId']))
                file.write('stationName: {}\n'.format(metadata['name']))
                file.write('location: {}\n'.format(metadata['location']))
                file.write('timebase: {}\n'.format(metadata['timebase']))
                file.write('regionId: {}\n'.format(metadata['regionId']))


if __name__ == "__main__":
    for index, station_id in enumerate(config.ims_mapping[config.ims_scrape_config['left_bound']:config.ims_scrape_config['right_bound']]):

        print('processing station: {}'.format(index + 1 + config.ims_scrape_config['left_bound']))

        obj=IMS_Scrapper_obj(index=index + 1 + config.ims_scrape_config['left_bound'], station_id=station_id,
                         _from=config.ims_scrape_config['_from'], _to=config.ims_scrape_config['_to'])

        if 'DOWNLOAD' in SELECTOR:
            obj.download_from_ims()

        if 'UPLOAD' in SELECTOR:
            obj.upload_files_to_gcs()
