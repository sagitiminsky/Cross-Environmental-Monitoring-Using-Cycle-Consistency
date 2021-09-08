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
import threading
import zipfile
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as dt_delta
import numpy as np
from CellEnMon.libs.power_law.power_law import PowerLaw


class DME_Scrapper_obj:

    def __init__(self, mock=None):
        self.chrome_options = Options()
        # self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.delay = 60
        self.selector = '//*[@id="btnExportByFilter"]'
        self.xpaths = config.xpaths
        self.root_download = config.download_path
        self.root_data_files = config.dme_root_files
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=self.chrome_options)

        if not os.path.isdir(self.root_data_files):
            os.makedirs(self.root_data_files)

        # clear old
        self.delete_prev_from_downloads_if_poss()
        self.delete_prev_data_files_if_poss()

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
        if config.dme_scrape_config['link_objects']['link_id']:
            link_names = config.dme_scrape_config['link_objects']['link_id']
            self.extract_merge_save_csv(self.download_zip_files_wrapper(link_names))
        else:
            self.extract_merge_save_csv(self.download_zip_files_wrapper())

    def create_merged_df_dict(self, metadata_df):
        d = {}
        for index, metadata_row in metadata_df.iterrows():
            link_name = metadata_row['link_id']
            if link_name not in d:
                d[link_name] = {
                    'data': pd.DataFrame(),
                    'frequency': '',
                    'polarization': '',
                    'length': '',
                    'txsite_longitude': '',
                    'txsite_latitude': '',
                    'rxsite_longitude': '',
                    'rxsite_latitude': ''

                }

        if not d:
            raise Exception("no files in zip")

        return d

    def is_different(self, new_param, link_name, link_dict, key):
        if link_dict[key] and new_param != link_dict[key]:
            raise ValueError(
                'link_name:{} current param: {} of type: {} is different from:{}'.format(link_name, link_dict[key], key,
                                                                                         new_param))
        return new_param

    def extract_merge_save_csv(self, file_paths):
        print("starting extraction...")
        merged_df_dict = {}
        for data_path, metadata_path in zip(file_paths['data_paths'], file_paths['metadata_paths']):
            zip_file_object = zipfile.ZipFile(data_path, 'r')
            metadata_df = pd.read_csv(metadata_path)
            merged_df_dict = {**self.create_merged_df_dict(metadata_df=metadata_df), **merged_df_dict}

            for zip_file_name, (index, metadata_row) in zip(zip_file_object.namelist(),
                                                            metadata_df.iterrows()):

                file_name = metadata_row['carrier'] + '_' + metadata_row['link_id'] + '.txt'
                link_name = metadata_row['link_id']

                try:
                    bytes = zip_file_object.open(zip_file_name).read()
                    add_df = pd.read_csv(io.StringIO(bytes.decode('utf-8')), sep=',')

                    # zero-level
                    # median = np.median((-1) * add_df['RFInputPower'])

                    # #preprocessing - turn link frequency from MHz to GHz - because PowerLaw is in GHz
                    # power_law = PowerLaw(frequency=metadata_row['frequency'] / 1000,  # Link Frequency [MHz]
                    #                      polarization=metadata_row['polarization'],
                    #                      L=metadata_row['length'])

                    # # todo: 'RFInputPower' is only good for one type of link
                    # add_df['rain'] = power_law.basic_attinuation_to_rain_multiple(
                    #     (-1) * add_df['RFInputPower'] - median)

                    if 'DEBUG' in os.environ:
                        print('DEBUG: link id is: {} median is:{}'.format(link_name, median))

                    merged_df_dict[link_name]['data'] = merged_df_dict[link_name]['data'].append(add_df)

                    for metadata_feature in config.dme_metadata:
                        merged_df_dict[link_name][metadata_feature] = self.is_different(metadata_row[metadata_feature],
                                                                                        link_name=link_name,
                                                                                        link_dict=merged_df_dict[
                                                                                            link_name],
                                                                                        key=metadata_feature)


                except KeyError:
                    print('exist in metadata, but does not exist in data:{}, feature:{}'.format(file_name,
                                                                                                metadata_feature))

        print('extraction done.')
        print('starting csv save...')

        for link_name in merged_df_dict:

            try:
                link_file_name = f"{config.dme_root_files + link_name}_frequency[GHz]:_{str(float(merged_df_dict[link_name]['frequency']) / 1000)}_polarization:_{merged_df_dict[link_name]['polarization']}_length[Km]:_{str(merged_df_dict[link_name]['length'])}_txsite-longitude:_{str(merged_df_dict[link_name]['txsite_longitude'])}_txsite-latitude:_{str(merged_df_dict[link_name]['txsite_latitude'])}_rxsite-longitude:_{str(merged_df_dict[link_name]['rxsite_longitude'])}_rxsite-latitude:_{str(merged_df_dict[link_name]['rxsite_latitude'])}_.csv"

                self.preprocess_df(merged_df_dict[link_name]['data']).to_csv(link_file_name, mode='a', index=False)

                print("file saved to {}".format(link_file_name))

            except ValueError:
                print('frequecy missing for this file: {}'.format(link_name))

    print('csv done.')

    def preprocess_df(self, df):
        # order by time
        df = df.sort_values(by='Time')
        return df

    def get_link_config(self, link_name):
        return {'link_name': link_name}

    def background_task(self, prev_number_of_files, delta):
        if len(os.listdir(self.root_download)) - prev_number_of_files == delta:
            return

    def download_data(self, start_day, end_day=None):
        try:

            # download metadata
            self.browser.find_element_by_xpath(self.xpaths['xpath_metadata_download']).click()
            WebDriverWait(self.browser, self.delay).until(
                EC.element_to_be_clickable((By.XPATH, self.xpaths['xpath_metadata_download'])))

            time.sleep(1)

            # download data
            self.browser.find_element_by_xpath(self.xpaths['xpath_download']).click()
            WebDriverWait(self.browser, self.delay).until(
                EC.element_to_be_clickable((By.XPATH, self.xpaths['xpath_download'])))

        except TimeoutException:  # rows do not exist
            if not end_day:
                print('The following day: {} failed'.format(start_day))
                pass
            else:
                raise TimeoutException("The following range: {} - {} failed ".format(start_day, end_day))


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
            filter=self.browser.find_element_by_xpath(element_xpath['xpath_filter_range'])
            filter.click()
            filter.send_keys(select['value_range'])




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

    def download_zip_files_wrapper(self, link_names=None):

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
                self.download_data(start_day=start_day, end_day=end_day)

            data_paths = [self.root_download + f for f in os.listdir(self.root_download) if '.zip' in f and 'cldb' in f]
            metadata_paths = [self.root_download + f for f in os.listdir(self.root_download) if
                              '.csv' in f and 'cldb' in f]

        data_paths.sort(key=os.path.getmtime)
        metadata_paths.sort(key=os.path.getmtime)

        return {
            'data_paths': data_paths,
            'metadata_paths': metadata_paths
        }

    def log_in(self, browser):
        remember_me_xpth = '/html/body/div/form/div[4]/label/input'
        submit_button = '/html/body/div/form/div[3]/button'

        username = browser.find_element_by_name("username")
        password = browser.find_element_by_name("password")

        username.send_keys(config.dme_scrape_config['username'])
        password.send_keys(config.dme_scrape_config['password'])

        browser.find_element_by_xpath(remember_me_xpth).click()
        browser.find_element_by_xpath(submit_button).click()

    def delete_prev_data_files_if_poss(self):
        'deletes from local directory'
        for file in os.listdir(self.root_data_files):
            try:
                os.remove(self.root_data_files + file)
            except FileNotFoundError:
                pass

    def delete_prev_from_downloads_if_poss(self):
        'Deletes from Downloads'
        for file in os.listdir(self.root_download):
            if 'cldb' in file or 'export' in file:
                try:
                    os.remove(self.root_download + file)
                except PermissionError:
                    shutil.rmtree(self.root_download + file)
                except Exception:
                    raise Exception('unable to remove file : {}'.format(self.root_download + file))


DME_Scrapper_obj().scrape()
