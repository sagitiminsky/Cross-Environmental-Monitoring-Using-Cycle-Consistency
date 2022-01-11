---
id: 495ai
name: Omnisol (dme) scrapper
file_version: 1.0.2
app_version: 0.7.1-2
file_blobs:
  CellEnMon/libs/scrappers/dme_scrapper/scrapper.py: c03e64ea9e20d10eeec96eed0acdfa1ee27af37a
---

Hi and welcome to the Omnisol (dme) scrapper. I will quickly explain how to extract information for the Omnisol Dailey Measurement Explorer, or in short dme.

TL;DR python3 `📄 CellEnMon/libs/scrappers/dme_scrapper/scrapper.py`

I've created a short `📄 CellEnMon/libs/scrappers/dme_scrapper/requirements.txt` requirements file, in case you do not want to install torch for example, you are only intrested in running this script.

<br/>

The `📄 CellEnMon/config.py` is basically the controller for this. It defines the criteria for the scraping.
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/dme_scrapper/scrapper.py
```python
⬜ 1      import time
🟩 2      import CellEnMon.config as config
⬜ 3      import os
⬜ 4      import shutil
⬜ 5      import io
```

<br/>

The script serves multiple purpuses:

<br/>

you can select

DOWNLOAD - only to Download the files into your Downloads folder  
EXTRACT - extract the downloaded files from the Downloads folder and merge all the information based on the station  
UPLOAD - upload the extracted information to gcp
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/dme_scrapper/scrapper.py
```python
⬜ 22     from CellEnMon.libs.power_law.power_law import PowerLaw
⬜ 23     from google.cloud import storage
⬜ 24     
🟩 25     SELECTOR = ['EXTRACT', 'UPLOAD'] # DOWNLOAD | EXTRACT | UPLOAD
⬜ 26     
⬜ 27     
⬜ 28     ## Setting credentials using the downloaded JSON file
```

<br/>

If you will try to upload the files, make sure to have the proper creds to do so
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/dme_scrapper/scrapper.py
```python
⬜ 26     
⬜ 27     
⬜ 28     ## Setting credentials using the downloaded JSON file
🟩 29     if "UPLOAD" in SELECTOR:
🟩 30         path = config.bucket_creds
🟩 31         client = storage.Client.from_service_account_json(json_credentials_path=path)
🟩 32     
⬜ 33     class DME_Scrapper_obj:
⬜ 34     
⬜ 35         def __init__(self, mock=None):
```

<br/>

One of the problems I encoutered is the fact the that "Download Metadata" button, does not work. So I had to manually export the table from dme. I did it by pressing the "End" option using the script, so that it will bring to the end of the table - so that I could download the information.
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/dme_scrapper/scrapper.py
```python
⬜ 238            if len(os.listdir(self.root_download)) - prev_number_of_files == delta:
⬜ 239                return
⬜ 240    
🟩 241        def download_data(self, link_name,start_day, end_day=None):
🟩 242            try:
🟩 243    
🟩 244                # download data
🟩 245                self.browser.find_element_by_xpath(self.xpaths['xpath_download']).click()
🟩 246                WebDriverWait(self.browser, self.delay).until(
🟩 247                    EC.element_to_be_clickable((By.XPATH, self.xpaths['xpath_download'])))
🟩 248    
🟩 249                time.sleep(1)
🟩 250    
🟩 251                # download metadata
🟩 252                ActionChains(self.browser).context_click(
🟩 253                    self.browser.find_element_by_xpath('//*[@id="dailies"]/div/div[2]/div[1]/div[3]')).send_keys(Keys.END).perform()
🟩 254    
🟩 255                ActionChains(self.browser).context_click(
🟩 256                    self.browser.find_element_by_xpath('//*[@id="dailies"]/div/div[2]/div[1]/div[3]')).perform()
🟩 257                self.browser.find_element_by_xpath('//*[ @ id = "dailies"]/div/div[6]/div/div/div[5]/span[2]').click()
🟩 258    
🟩 259                self.browser.find_element_by_xpath(self.xpaths['xpath_metadata_download']).click()
⬜ 260    
⬜ 261    
⬜ 262    
```

<br/>

Meta data is going to be saved into the file's name
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 CellEnMon/libs/scrappers/dme_scrapper/scrapper.py
```python
⬜ 211                if valid:
⬜ 212                    # link_frequency=merged_df_dict[link_name]['frequency'][0]
⬜ 213                    # link_polarization=merged_df_dict[link_name]['polarization'][0]
⬜ 214                    # link_length=merged_df_dict[link_name]['length'][0]
⬜ 215                    link_tx_longitude=merged_df_dict[link_name]['tx_longitude'][0]
⬜ 216                    link_tx_latitude=merged_df_dict[link_name]['tx_latitude'][0]
⬜ 217                    link_rx_longitude=merged_df_dict[link_name]['rx_longitude'][0]
⬜ 218                    link_rx_latitude=merged_df_dict[link_name]['rx_latitude'][0]
⬜ 219                    tx_name,rx_name=link_name.split("-")
🟩 220                    metadata=f"{tx_name}-{link_tx_latitude}-{link_tx_longitude}-{rx_name}-{link_rx_latitude}-{link_rx_longitude}"
⬜ 221                    link_file_name= f"{config.dme_root_files}/raw/{metadata}.csv"
⬜ 222    
⬜ 223                    try:
```

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBQ2VsbEVuTW9uLVJlc2VhcmNoJTNBJTNBc2FnaXRpbWluc2t5/docs/495ai).