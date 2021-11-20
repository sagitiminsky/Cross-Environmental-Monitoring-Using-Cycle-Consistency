---
id: 8L_U1
name: IMS Services Subscription Guide
file_version: 1.0.2
app_version: 0.6.6-2
file_blobs:
  CellEnMon/config.py: 65bffe1061bb4439484b680480f717537efe53f6
  CellEnMon/libs/scrappers/ims_scrapper/scrapper.py: 7b69bed4fc6c6f9157d22d8d34f56634c1ec18c2
---

TLDR: send go to [https://ims.gov.il/he/ObservationDataAPI](https://ims.gov.il/he/ObservationDataAPI) you will find the API documenation and you'll be able to fill in the following form [https://ims.gov.il/sites/default/files/docs/terms\_0.pdf](https://ims.gov.il/sites/default/files/docs/terms_0.pdf) and send it back to ims@ims.gov.il.

After receiving the approval from the ims you will be provided with an API Token, if you want, you can keep that in secret - put that in your gitignore file.

You can use the following script:

<br/>

Here is ims station mapping
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/config.py
```python
⬜ 72     ims_root_files = f"datasets/ims/raw/{start_date_str_rep}_{end_date_str_rep}"  # DD/MM/YYYY
⬜ 73     ims_root_values = 'datasets/ims/processed'
⬜ 74     ims_token = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
🟩 75     ims_mapping = [
🟩 76         '241',
🟩 77         '348',
🟩 78         '202',
🟩 79         '10',
🟩 80         '106',
🟩 81         '73',
🟩 82         '353',
🟩 83         '343',
🟩 84         '62',
🟩 85         '269',
🟩 86         '123',
🟩 87         '227',
🟩 88         '205',
🟩 89         '233',
🟩 90         '6',
🟩 91         '99',
🟩 92         '78',
🟩 93         '46',
🟩 94         '115',
🟩 95         '2',
🟩 96         '41',
🟩 97         '43',
🟩 98         '42',
🟩 99         '186',
🟩 100        '13',
🟩 101        '8',
🟩 102        '11',
🟩 103        '355',
🟩 104        '44',
🟩 105        '264',
🟩 106        '67',
🟩 107        '16',
🟩 108        '45',
🟩 109        '263',
🟩 110        '380',
🟩 111        '224',
🟩 112        '46',
🟩 113        '206',
🟩 114        '366',
🟩 115        '107',
🟩 116        '20',
🟩 117        '90',
🟩 118        '275',
🟩 119        '270',
🟩 120        '21',
🟩 121        '178',
🟩 122        '54',
🟩 123        '30',
🟩 124        '24',
🟩 125        '124',
🟩 126        '259',
🟩 127        '74',
🟩 128        '228',
🟩 129        '121',
🟩 130        '188',
🟩 131        '23',
🟩 132        '218',
🟩 133        '22',
🟩 134        '274',
🟩 135        '75',
🟩 136        '25',
🟩 137        '77',
🟩 138        '82',
🟩 139        '208',
🟩 140        '236',
🟩 141        '210',
🟩 142        '79',
🟩 143        '211',
🟩 144        '350',
🟩 145        '28',
🟩 146        '58',
🟩 147        '59',
🟩 148        '29',
🟩 149        '349',
🟩 150        '112',
🟩 151        '65',
🟩 152        '98',
🟩 153        '338',
🟩 154        '271',
🟩 155        '33',
🟩 156        '379',
🟩 157        '207',
🟩 158        '232',
🟩 159        '36',
🟩 160        '64'
🟩 161    ]
⬜ 162    
⬜ 163    ims_scrape_config = {
⬜ 164        '_from': f"{add_days_to_date(date['value'])['str_rep']}",  # MM/DD/YYYY
```

<br/>

Let's define the authentication token from what we've got from the IMS
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/ims_scrapper/scrapper.py
```python
⬜ 8      
⬜ 9      url = "https://api.ims.gov.il/v1/envista/stations/64/data?from=2019/12/01&to=2020/01/01"
⬜ 10     
🟩 11     headers = {
🟩 12         'Authorization': 'ApiToken ' + config.ims_token
🟩 13     }
⬜ 14     
⬜ 15     
⬜ 16     ## Setting credentials using the downloaded JSON file
```

<br/>

The endpoints we are about to use
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/ims_scrapper/scrapper.py
```python
⬜ 32             self._from = _from
⬜ 33             self._to = _to
⬜ 34             self.root = config.ims_root_files
🟩 35             self.station_meta_data = f"https://api.ims.gov.il/v1/envista/stations/{station_id}"
🟩 36             self.station_data = f"https://api.ims.gov.il/v1/envista/stations/{station_id}/data/?from={_from}&to={_to}"
⬜ 37             self.bucket = client.get_bucket('cell_en_mon')
⬜ 38     
⬜ 39         def upload_files_to_gcs(self):
```

<br/>

We will validate the indeed we get a 200 response from the api
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/ims_scrapper/scrapper.py
```python
⬜ 49                         print(f'Uploaded file:{file} failed with the following exception:{e}!')
⬜ 50     
⬜ 51         def download_from_ims(self):
🟩 52             metadata_response = requests.request("GET", self.station_meta_data, headers=headers)
🟩 53             data_response = requests.request("GET", self.station_data, headers=headers)
⬜ 54     
⬜ 55             if metadata_response.status_code != 200 or data_response.status_code != 200:
⬜ 56                 print("station id: {} , metadata respose: {} , data response: {}".format(self.station_id,
```

<br/>

load the metadata and data locally
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/ims_scrapper/scrapper.py
```python
⬜ 57                                                                                          metadata_response.status_code,
⬜ 58                                                                                          data_response.status_code))
⬜ 59     
🟩 60             else:
🟩 61                 metadata = json.loads(metadata_response.text.encode('utf8'))
🟩 62                 data = json.loads(data_response.text.encode('utf8'))
🟩 63     
⬜ 64                 folder = "{}-{}-{}".format(self.index, metadata['stationId'], metadata['name'])
⬜ 65                 if not os.path.exists(self.root + '/' + folder):
⬜ 66                     os.makedirs(self.root + '/' + folder)
```

<br/>

save metadata and data
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/ims_scrapper/scrapper.py
```python
⬜ 71                         os.remove(self.root + '/' + 'data.csv')
⬜ 72                     except FileNotFoundError:
⬜ 73                         pass
🟩 74     
🟩 75                 pd.DataFrame(metadata['monitors']).to_csv(self.root + '/' + folder + '/' + "monitors.csv", index=False)
🟩 76                 pd.DataFrame(data['data']).to_csv(self.root + '/' + folder + '/' + "data.csv", index=False)
🟩 77                 with open(self.root + '/' + folder + '/' + "metadata.txt", 'w') as file:
🟩 78                     file.write('stationId: {}\n'.format(metadata['stationId']))
🟩 79                     file.write('stationName: {}\n'.format(metadata['name']))
🟩 80                     file.write('location: {}\n'.format(metadata['location']))
🟩 81                     file.write('timebase: {}\n'.format(metadata['timebase']))
🟩 82                     file.write('regionId: {}\n'.format(metadata['regionId']))
⬜ 83     
⬜ 84     #
⬜ 85     if __name__ == "__main__":
```

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBQ2VsbEVuTW9uLVJlc2VhcmNoJTNBJTNBc2FnaXRpbWluc2t5/docs/8L_U1).