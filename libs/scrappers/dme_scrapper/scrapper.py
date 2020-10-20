import time
import config as config
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


class DME_Scrapper_obj:

    def __init__(self, mock=None):
        self.chrome_options = Options()
        # self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.delay = 30
        self.selector = '//*[@id="btnExportByFilter"]'
        self.xpaths = {
            'xpath_download': '//*[@id="btnExportByFilter"]',
            'xpath_metadata_download': '/html/body/div[3]/div/div[7]/div/div/div[1]/span[2]',
            'link_id': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[6]/div/div[1]',
                'xpath_select': '',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[6]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[6]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'
            },
            'date': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[1]/div/div[1]',
                'xpath_select': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[1]/div/div[3]/div/div[2]/div/div/div[1]/select[1]',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[1]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[1]/div/input',
                'xpath_filter_range': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[1]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[2]/div/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[1]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'
            },
            'measurement_type': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[1]',
                'xpath_select_all': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[1]/div[2]/div[1]/label/span',
                'xpath_hc_radio_sink': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[1]/div[2]/div[2]/div/div/div[1]/label',
                'xpath_hc_radio_source': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[1]/div[2]/div[2]/div/div/div[2]/label',
                'xpath_tn_rfinputpower': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[1]/div[2]/div[2]/div/div/div[4]/label/span',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'

            },
            'sampling_period[sec]': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[4]/div/div[1]/span[2]',
                'input_box': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[4]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div/input',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[4]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[4]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'

            },
            'rx_site_longitude': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[20]/div/div[1]/span[2]',
                'xpath_select': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[20]/div/div[3]/div/div[2]/div/div/div[1]/select[1]',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[20]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[1]/input',
                'xpath_filter_range': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[20]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[2]/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[20]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'
            },
            'rx_site_latitude': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[21]/div/div[1]/span[2]',
                'xpath_select': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[21]/div/div[3]/div/div[2]/div/div/div[1]/select[1]',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[21]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[1]/input',
                'xpath_filter_range': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[21]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[2]/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[21]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'
            },
            'tx_site_longitude': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[15]/div/div[1]/span[2]',
                'xpath_select': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[15]/div/div[3]/div/div[2]/div/div/div[1]/select[1]',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[15]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[1]/input',
                'xpath_filter_range': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[15]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[2]/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[15]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[2]/input'
            },
            'tx_site_latitude': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[16]/div/div[1]/span[2]',
                'xpath_select': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[16]/div/div[3]/div/div[2]/div/div/div[1]/select[1]',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[16]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[1]/input',
                'xpath_filter_range': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[16]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[2]/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[16]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'
            },
            'link_frequency[mhz]': {
                'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[8]/div/div[1]/span[2]',
                'xpath_select': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[8]/div/div[3]/div/div[2]/div/div/div[1]/select[1]',
                'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[8]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[1]/input',
                'xpath_filter_range': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[8]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[2]/input',
                'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[8]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'
            }

        }
        self.root_download = '/Users/sagit/Downloads/'
        self.root_data_files = config.dme_scrape_config['path_to_data_files']
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=self.chrome_options)
        self.accepted_link_type = ['SOURCE', 'SINK']

        # clear old files from downloads
        self.delete_prev_from_downloads_if_poss()
        self.delete_prev_data_files_if_poss()

        # log in
        self.browser.get(config.dme_scrape_config['url'])
        self.log_in(self.browser)

        # time frame
        self.start_datetime = self.convert_to_datetime_and_add_delta_days(
            config.dme_scrape_config['link_objects']['date']['value'])
        self.end_datetime = self.convert_to_datetime_and_add_delta_days(
            config.dme_scrape_config['link_objects']['date']['value_range'])
        self.time_frame = (self.end_datetime['datetime_rep'] - self.start_datetime['datetime_rep']).days + 1

        # accept alert
        time.sleep(3)
        alert = self.browser.switch_to_alert()
        alert.dismiss()

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
            for link_name in config.dme_scrape_config['link_objects']['link_id']:
                self.extract_merge_save_csv(self.download_zip_files(link_name))
        else:
            self.extract_merge_save_csv(self.download_zip_files())

    def create_merged_df_dict(self, file_names,metadata_df):
        d = {}
        for file_name,(index,metadata_row) in zip(file_names,metadata_df.iterrows()):
            link_name = metadata_row['Link ID']
            if link_name not in d:
                # todo: iterate over self.accepted_link_types
                d[link_name] = {
                    'data': pd.DataFrame(),
                    'frequency': '',
                    'polarization': '',
                    'L': ''
                }

        if not d:
            raise Exception("no files in zip")

        return d

    def is_different(self, new_param,link_name, link_dict, key):
        if link_dict[key] and new_param != link_dict[key]:
            raise ValueError(
                'link_name:{} current param: {} of type: {} is different from:{}'.format(link_name,link_dict[key], key, new_param))
        return new_param

    def extract_merge_save_csv(self, file_paths):
        merged_df_dict = {}
        for data_path, metadata_path in zip(file_paths['data_paths'], file_paths['metadata_paths']):
            zip_file_object = zipfile.ZipFile(data_path, 'r')
            metadata_df=pd.read_csv(metadata_path)
            merged_df_dict = {**self.create_merged_df_dict(zip_file_object.namelist(),metadata_df=metadata_df), **merged_df_dict}

            for file_name, (index, metadata_row) in zip(zip_file_object.namelist(),
                                                        metadata_df.iterrows()):

                link_name = metadata_row['Link ID']


                bytes = zip_file_object.open(file_name).read()
                add_df = pd.read_csv(io.StringIO(bytes.decode('utf-8')), sep=',')
                merged_df_dict[link_name]['data'] = merged_df_dict[link_name]['data'].append(add_df)
                merged_df_dict[link_name]['frequency'] = self.is_different(metadata_row['Link Frequency [MHz]'],
                                                                           link_name=link_name,
                                                                           link_dict=merged_df_dict[link_name],
                                                                           key='frequency')
                merged_df_dict[link_name]['polarization'] = self.is_different(metadata_row['Link Polarization'],
                                                                              link_name=link_name,
                                                                              link_dict=merged_df_dict[link_name],
                                                                              key='polarization')
                merged_df_dict[link_name]['L'] = self.is_different(metadata_row['Link Length (KM)'],
                                                                   link_name=link_name,
                                                                   link_dict=merged_df_dict[link_name],
                                                                   key='L')

        for link_name in merged_df_dict:
            link_file_name = config.dme_scrape_config['path_to_data_files'] + link_name + '_' + \
                             'frequency[MHz]:' + str(merged_df_dict[link_name]['frequency']) + '_' + \
                             'polarization:' + merged_df_dict[link_name]['polarization'] + '_' + \
                             'L[Km]:' + str(merged_df_dict[link_name]['L']) + '.csv'

            self.preprocess_df(merged_df_dict[link_name]['data']).to_csv(link_file_name, mode='a', index=False)

            print("file saved to {}".format(link_file_name))

    def preprocess_df(self, df):
        # order by time
        df = df.sort_values(by='Time')
        return df

    def get_link_config(self, link_name):
        return {'link_name': link_name}

    def background_task(self, prev_number_of_files, delta):
        if len(os.listdir(self.root_download)) - prev_number_of_files == delta:
            return

    def ranged_filter(self, mux):
        link_obj = config.dme_scrape_config['link_objects']
        select = link_obj[mux]
        select_value = select['value']
        element_xpath = self.xpaths[mux]
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])

        if mux == 'date':
            pass

        elif mux == 'tx_site_longitude' or mux == 'tx_site_latitude' or mux == 'rx_site_longitude' or mux == 'rx_site_latitude' or mux == 'link_frequency[mhz]':
            Select(self.browser.find_element_by_xpath(element_xpath['xpath_select'])).select_by_visible_text(
                select['select'])
            filter.send_keys(select_value)
        else:
            raise ValueError("mux type is undefined {}".format(mux))

        if select['select'] == 'In range':
            select_value_range = select['value_range']

            if mux == 'date':

                # download
                print('starting download...')

                day_iter = self.start_datetime
                counter = 0
                while day_iter['datetime_rep'] <= self.end_datetime['datetime_rep']:
                    print('download day #{}/{}, date:{}'.format(counter + 1, self.time_frame, day_iter['str_rep']))
                    self.browser.find_element_by_xpath(element_xpath['xpath_filter']).click()
                    filter.send_keys(day_iter['str_rep'])
                    self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

                    day_iter = self.convert_to_datetime_and_add_delta_days(day_iter['dict_rep'], delta_days=1)
                    counter = counter + 1

                    time.sleep(15)
                    # download metadata
                    ActionChains(self.browser).context_click(
                        self.browser.find_element_by_xpath('//*[@id="dailies"]/div/div[2]/div[1]/div[3]')).perform()
                    self.browser.find_element_by_xpath('//*[@id="dailies"]/div/div[6]/div/div/div[5]/span[2]').click()
                    self.browser.find_element_by_xpath(self.xpaths['xpath_metadata_download']).click()

                    time.sleep(1)
                    # download data
                    self.browser.find_element_by_xpath(self.xpaths['xpath_download']).click()

            elif mux == 'tx_site_longitude' or mux == 'tx_site_latitude' or mux == 'rx_site_longitude' or mux == 'rx_site_latitude':
                Select(self.browser.find_element_by_xpath(element_xpath['xpath_select'])).select_by_visible_text(
                    select['select'])
                filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter_range'])
                filter.send_keys(select_value_range)
            else:
                raise ValueError("mux type is undefined {}".format(mux))

        self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

    def check_boxes(self):
        link_obj = config.dme_scrape_config['link_objects']
        element_xpath = self.xpaths['measurement_type']
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        self.browser.find_element_by_xpath(element_xpath['xpath_select_all']).click()

        if 'HC_RADIO_SINK' in link_obj['measurement_type']:
            self.browser.find_element_by_xpath(element_xpath['xpath_hc_radio_sink']).click()
        if 'HC_RADIO_SOURCE' in link_obj['measurement_type']:
            self.browser.find_element_by_xpath(element_xpath['xpath_hc_radio_source']).click()
        if 'TN_RFInputPower' in link_obj['measurement_type']:
            self.browser.find_element_by_xpath(element_xpath['xpath_tn_rfinputpower']).click()

    def input_box(self, input_type):
        element_xpath = self.xpaths[input_type]
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])
        filter.send_keys(config.dme_scrape_config['link_objects'][input_type])
        self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

    def download_zip_files(self, link_name=None):
        # link id
        if link_name:
            element_xpath = self.xpaths['link_id']
            self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
            filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])
            filter.send_keys(link_name)
            self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

        # measurement type
        self.check_boxes()

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
        self.input_box('sampling_period[sec]')

        # Link frequency[MHz]
        self.ranged_filter('link_frequency[mhz]')

        # date
        self.ranged_filter('date')

        time.sleep(30)

        data_paths = [self.root_download + f for f in os.listdir(self.root_download) if 'cldb' in f]
        metadata_paths = [self.root_download + f for f in os.listdir(self.root_download) if 'export' in f]

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

    def wait_to_element_load(self, xpath):
        try:
            # wait for data to be loaded
            WebDriverWait(self.browser, self.delay).until(EC.presence_of_element_located((By.XPATH, xpath)))

        except TimeoutException:
            print('Loading took too much time!')

        finally:
            self.browser.quit()

    def delete_prev_data_files_if_poss(self):
        for file in os.listdir(self.root_data_files):
            try:
                os.remove(self.root_data_files + file)
            except FileNotFoundError:
                pass

    def delete_prev_from_downloads_if_poss(self):
        for file in os.listdir(self.root_download):
            if 'cldb' in file or 'export' in file:
                try:
                    os.remove(self.root_download + file)
                except PermissionError:
                    shutil.rmtree(self.root_download + file)
                except Exception:
                    raise Exception('unable to remove file : {}'.format(self.root_download + file))


DME_Scrapper_obj().scrape()
