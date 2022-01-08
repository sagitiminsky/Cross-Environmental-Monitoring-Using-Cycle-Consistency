import time
import CellEnMon.config as config
import os
import shutil
import io
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import threading
import zipfile
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as dt_delta
import numpy as np
from CellEnMon.libs.power_law.power_law import PowerLaw
from google.cloud import storage

SELECTOR = ['EXTRACT', 'UPLOAD'] # DOWNLOAD | EXTRACT | UPLOAD


## Setting credentials using the downloaded JSON file
if "UPLOAD" in SELECTOR:
    path = config.bucket_creds
    client = storage.Client.from_service_account_json(json_credentials_path=path)

class DME_Scrapper_obj:

    def __init__(self, mock=None):
        self.chrome_options = Options()
        # self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.delay = 10
        self.selector = '//*[@id="btnExportByFilter"]'
        self.xpaths = config.xpaths
        self.root_download = config.download_path
        self.root_data_files = config.dme_root_files
        if "UPLOAD" in SELECTOR:
            self.bucket = client.get_bucket('cell_en_mon')

        if not os.path.isdir(self.root_data_files):
            os.makedirs(self.root_data_files)

        if 'DOWNLOAD' in SELECTOR:
            self.browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=self.chrome_options)

            # clear old
            self.delete_prev_from_downloads_if_poss()
            self.delete_prev_data_files_if_poss(self.root_data_files)

            # log in
            self.browser.get(config.dme_scrape_config['url'])
            self.log_in(self.browser)

        # time frame
        self.start_datetime = self.convert_to_datetime_and_add_delta_days(config.date['value'])
        self.end_datetime = self.convert_to_datetime_and_add_delta_days(config.date['value_range'])
        self.time_frame = (self.end_datetime['datetime_rep'] - self.start_datetime['datetime_rep']).days + 1

        # accept alert
        time.sleep(1)
        # alert = self.browser.switch_to_alert()
        # alert.dismiss()

    def parse_date(self, d):
        return d['mm'] + '/' + d['dd'] + '/' + d['yyyy'][-2:]

    def convert_to_datetime_and_add_delta_days(self, date, delta_days=0):
        res = dt.strptime(self.parse_date(date), "%m/%d/%y") + dt_delta(days=delta_days)
        return {
            'datetime_rep': res,
            'str_rep': f"{str(res.day).zfill(2) + str(res.month).zfill(2) + str(res.year)}",
            'dict_rep': {
                'dd': str(res.day).zfill(2),
                'mm': str(res.month).zfill(2),
                'yyyy': str(res.year)
            }
        }

    def scrape(self):

        if 'DOWNLOAD' in SELECTOR:
            self.download_zip_files_wrapper()

        if 'EXTRACT' in SELECTOR:
            self.extract_merge_save_csv()

        if 'UPLOAD' in SELECTOR:
            self.upload_files_to_gcs()



    def create_merged_df_dict(self,metadata_df):
        return {
                'data': pd.DataFrame(),
                # 'frequency': metadata_df[config.dme_metadata['frequency']],
                # 'polarization': metadata_df[config.dme_metadata['polarization']],
                # 'length': metadata_df[config.dme_metadata['length']],
                'tx_longitude': metadata_df[config.dme_metadata['tx_longitude']],
                'tx_latitude': metadata_df[config.dme_metadata['tx_latitude']],
                'rx_longitude': metadata_df[config.dme_metadata['rx_longitude']],
                'rx_latitude': metadata_df[config.dme_metadata['rx_latitude']]
            }


    def is_different(self, new_param, link_name, link_dict, key):
        if link_dict[key] and new_param != link_dict[key]:
            raise ValueError(
                'link_name:{} current param: {} of type: {} is different from:{}'.format(link_name, link_dict[key], key,
                                                                                         new_param))
        return new_param

    def upload_files_to_gcs(self):
        root=f"{config.dme_root_files}/raw"
        for file in os.listdir(root):
            blob = self.bucket.blob(f'dme/{config.start_date_str_rep}-{config.end_date_str_rep}/raw/{file}')
            try:
                with open(f"{root}/{file}", 'rb') as f:
                    blob.upload_from_file(f)
                print(f'Uploaded file:{file} succesfully !')
            except Exception:
                print(f'Uploaded file:{file} failed !')


    def extract_merge_save_csv(self):

        data_paths = [f"{config.download_path}/{f}" for f in os.listdir(self.root_download) if
                      '.zip' in f and 'cldb' in f]
        metadata_paths = [f"{config.download_path}/{f}" for f in os.listdir(self.root_download) if
                          '.csv' in f and 'export' in f]

        data_paths.sort(key=os.path.getmtime)
        metadata_paths.sort(key=os.path.getmtime)
        paths="CellEnMon/libs/scrappers/dme_scrapper/paths/"
        with open(f'{paths}/data_paths.txt', 'w') as f:
            for item in data_paths:
                f.write("{}\n".format(item))

        with open(f'{paths}/metadata_paths.txt', 'w') as f:
            for item in metadata_paths:
                f.write("{}\n".format(item))

        with open(f'{paths}/data_paths.txt') as f1, open(f'{paths}/metadata_paths.txt') as f2:
            file_paths = {
                'data_paths': f1.readlines(),
                'metadata_paths': f2.readlines()
            }

        carrier = config.dme_metadata['carrier']
        id = config.dme_metadata['id']
        frequency = config.dme_metadata['frequency']
        polarization = config.dme_metadata['polarization']
        length = config.dme_metadata['length']
        tx_longitude=config.dme_metadata['tx_longitude']
        tx_latitude = config.dme_metadata['tx_latitude']
        rx_longitude = config.dme_metadata['rx_longitude']
        rx_latitude = config.dme_metadata['rx_latitude']

        print("starting extraction...")

        merged_df_dict = {}
        for data_path, metadata_path, link_name in zip(file_paths['data_paths'], file_paths['metadata_paths'], config.dme_scrape_config['link_objects']['link_id']):
            valid=True
            zip_file_object = zipfile.ZipFile(data_path.strip(), 'r')
            metadata_df = pd.read_csv(metadata_path.strip())

            try:
                merged_df_dict[link_name] = {**self.create_merged_df_dict(metadata_df=metadata_df), **merged_df_dict}
            except KeyError:
                print(f"Missing critical information for invalid link:{link_name}")
                valid=False
                continue



            for zip_file_name in zip_file_object.namelist(): #metadata_df.iterrows()):


                #TODO: after the fix this data should come from the metadatafile, which is now empyu
                #link_name = metadata_row[id]




                bytes = zip_file_object.open(zip_file_name).read()
                add_df = pd.read_csv(io.StringIO(bytes.decode('utf-8')), sep=',')

                # zero-level
                # median = np.median((-1) * add_df['RFInputPower'])

                # #preprocessing - turn link frequency from MHz to GHz - because PowerLaw is in GHz
                # power_law = PowerLaw(frequency=metadata_row['frequency'] / 1000,  # Link Frequency [MHz]
                #                      polarization=metadata_row['polarization'],
                #                      L=metadata_row['length'])

                # TODO: we should also consider translating to rain amout at the source
                # add_df['rain'] = power_law.basic_attinuation_to_rain_multiple(
                #     (-1) * add_df['RFInputPower'] - median)

                merged_df_dict[link_name]['data'] = merged_df_dict[link_name]['data'].append(add_df)


            if valid:
                # link_frequency=merged_df_dict[link_name]['frequency'][0]
                # link_polarization=merged_df_dict[link_name]['polarization'][0]
                # link_length=merged_df_dict[link_name]['length'][0]
                link_tx_longitude=merged_df_dict[link_name]['tx_longitude'][0]
                link_tx_latitude=merged_df_dict[link_name]['tx_latitude'][0]
                link_rx_longitude=merged_df_dict[link_name]['rx_longitude'][0]
                link_rx_latitude=merged_df_dict[link_name]['rx_latitude'][0]
                tx_name,rx_name=link_name.split("-")
                metadata=f"{tx_name}-{link_tx_latitude}-{link_tx_longitude}-{rx_name}-{link_rx_latitude}-{link_rx_longitude}"
                link_file_name= f"{config.dme_root_files}/processed/{metadata}.csv"

                try:
                    self.preprocess_df(merged_df_dict[link_name]['data']).to_csv(link_file_name, mode='a', index=False)
                    print("file saved to {}".format(link_file_name))
                except KeyError:
                    print(f"faild saving link:{link_name}")

    def preprocess_df(self, df):
        # order by time
        df = df.sort_values(by='Time')
        return df

    def get_link_config(self, link_name):
        return {'link_name': link_name}

    def background_task(self, prev_number_of_files, delta):
        if len(os.listdir(self.root_download)) - prev_number_of_files == delta:
            return

    def download_data(self, link_name,start_day, end_day=None):
        try:

            # download data
            self.browser.find_element_by_xpath(self.xpaths['xpath_download']).click()
            WebDriverWait(self.browser, self.delay).until(
                EC.element_to_be_clickable((By.XPATH, self.xpaths['xpath_download'])))

            time.sleep(1)

            # download metadata
            ActionChains(self.browser).context_click(
                self.browser.find_element_by_xpath('//*[@id="dailies"]/div/div[2]/div[1]/div[3]')).send_keys(Keys.END).perform()

            ActionChains(self.browser).context_click(
                self.browser.find_element_by_xpath('//*[@id="dailies"]/div/div[2]/div[1]/div[3]')).perform()
            self.browser.find_element_by_xpath('//*[ @ id = "dailies"]/div/div[6]/div/div/div[5]/span[2]').click()

            self.browser.find_element_by_xpath(self.xpaths['xpath_metadata_download']).click()




        except TimeoutException:  # rows do not exist
            if not end_day:
                print('The following day: {} failed'.format(start_day))
                pass
            else:
                print(f"The following range: {start_day} - {end_day} failed for link:{link_name}")

    def ranged_filter(self, mux):
        link_obj = config.dme_scrape_config['link_objects']
        select = link_obj[mux]
        select_value = select['value']
        element_xpath = self.xpaths[mux]
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()

        Select(self.browser.find_element_by_xpath(element_xpath['xpath_select'])).select_by_visible_text(
            select['select'])

        self.browser.find_element_by_xpath(element_xpath['xpath_filter']).send_keys(select_value)

        if select['select'] == 'In range':
            filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter_range'])
            filter.click()
            filter.send_keys(select['value_range'])

        # Each filter needs to be applied
        self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

    def check_boxes(self):
        link_obj = config.dme_scrape_config['link_objects']
        element_xpath = self.xpaths['measurement_type']
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        self.browser.find_element_by_xpath(element_xpath['xpath_select_all']).click()

        search_box = self.browser.find_element_by_xpath(element_xpath['search_box'])

        if 'TN_RFInputPower' in link_obj['measurement_type']:
            search_box.send_keys('TN_RFInputPower')

        self.browser.find_element_by_xpath(element_xpath['xpath_select_all']).click()

    def input_box(self, input_type):
        element_xpath = self.xpaths[input_type]
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])
        filter.send_keys(config.dme_scrape_config['link_objects'][input_type])
        self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

    def download_zip_files_wrapper(self):

        link_names = config.dme_scrape_config['link_objects']['link_id']
        data_paths = []
        metadata_paths = []

        start_day = self.start_datetime['str_rep']
        end_day = self.end_datetime['str_rep']

        # measurement type
        # self.check_boxes()

        # # tx site longitude
        # self.ranged_filter('tx_site_longitude')
        #
        # # tx site latitude
        # self.ranged_filter('tx_site_latitude')
        #
        # # rx site longitude
        # self.ranged_filter('rx_site_longitude')
        #
        # # rx site latitude
        # self.ranged_filter('rx_site_latitude')

        # sampling period description
        self.input_box('sampling_period[min]')

        # Link frequency[MHz]
        self.ranged_filter('link_frequency[mhz]')

        # data_precentage
        self.ranged_filter('data_precentage')

        # date
        self.ranged_filter('date')

        # links id
        element_xpath = self.xpaths['link_id']
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])

        if link_names:
            for link_name in link_names:
                print("current link is:{}".format(link_name))

                # link id
                self.browser.find_element_by_xpath(element_xpath['xpath_reset']).click()
                filter.send_keys(link_name)
                self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

                print('starting download range {}-{}...'.format(start_day, end_day))
                self.download_data(link_name=link_name, start_day=start_day, end_day=end_day )

        else:  # No need to look for link_id
            print('starting download range {}-{}...'.format(start_day, end_day))
            self.download_data(link_name='Nan',start_day=start_day, end_day=end_day)


    def log_in(self, browser):
        remember_me_xpth = '/html/body/div/form/div[4]/label/input'
        submit_button = '/html/body/div/form/div[3]/button'

        username = browser.find_element_by_name("username")
        password = browser.find_element_by_name("password")

        username.send_keys(config.dme_scrape_config['username'])
        password.send_keys(config.dme_scrape_config['password'])

        browser.find_element_by_xpath(remember_me_xpth).click()
        browser.find_element_by_xpath(submit_button).click()

    def delete_prev_data_files_if_poss(self, path):
        'deletes from local directory'
        try:
            for file in os.listdir(path):
                try:
                    os.remove(f"{path}/{file}")
                except FileNotFoundError:
                    print(f"throwing FileNotFoundError for:{path}/{file}")
        except FileNotFoundError:
            print(f"File not found error:{path}")

    def delete_prev_from_downloads_if_poss(self):
        'Deletes from Downloads'
        for file in os.listdir(self.root_download):
            if 'cldb' in file or 'export' in file:
                try:
                    os.remove(f"{self.root_download}/{file}")
                except PermissionError:
                    shutil.rmtree(f"{self.root_download}/{file}")
                except Exception:
                    raise Exception('unable to remove file : {}'.format(f"{self.root_download}/{file}"))



if __name__=="__main__":
    DME_Scrapper_obj().scrape()
