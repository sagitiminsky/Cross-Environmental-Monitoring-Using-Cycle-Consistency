import json
import requests
import pandas as pd
import CellEnMon.config as config
import os
from datetime import datetime as dt

url = "https://api.ims.gov.il/v1/envista/stations/64/data?from=2019/12/01&to=2020/01/01"

headers = {
    'Authorization': 'ApiToken ' + config.ims_token
}


class IMS_Scrapper_obj:
    def __init__(self, index, station_id, _from, _to):
        self.index = index
        self.station_id = station_id
        self._from = _from
        self._to = _to
        self.root = config.ims_root_files
        self.station_meta_data = f"https://api.ims.gov.il/v1/envista/stations/{station_id}"
        self.station_data = f"https://api.ims.gov.il/v1/envista/stations/{station_id}/data/?from={_from}&to={_to}"

    def save_metadata(self):
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

        IMS_Scrapper_obj(index=index + 1 + config.ims_scrape_config['left_bound'], station_id=station_id,
                         _from=config.ims_scrape_config['_from'], _to=config.ims_scrape_config['_to']).save_metadata()
