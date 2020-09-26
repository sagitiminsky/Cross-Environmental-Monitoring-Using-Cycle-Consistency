import time
import config as config
import os
import shutil
import io
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import threading
import zipfile
import pandas as pd
from datetime import datetime as dt


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

        }
        self.root_download = '/Users/sagit/Downloads/'
        self.root_data_files = config.dme_scrape_config['path_to_data_files']
        self.exclude_file = ['.DS_Store', '.localized', '.com.google.Chrome.tlwaEL']
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=self.chrome_options)

        # date validation
        value = self.parse_date(config.dme_scrape_config['link_objects']['date']['value']).strip()
        value_range = self.parse_date(config.dme_scrape_config['link_objects']['date']['value_range']).strip()
        date_select = config.dme_scrape_config['link_objects']['date']['select']

        if not value or date_select == 'In range' and dt.strptime(value, "%m/%d/%y") > dt.strptime(value_range,
                                                                                                   "%m/%d/%y"):
            raise ValueError('missing value in date, or value_range is earlier than value')

        self.covrage = (dt.strptime(value_range, "%m/%d/%y") - dt.strptime(value, "%m/%d/%y")).days + 1

        # clear old files from downloads
        self.delete_prev_from_downloads_if_poss()
        self.delete_prev_data_files_if_poss()

        # log in
        self.browser.get(config.dme_scrape_config['url'])
        self.log_in(self.browser)

        # accept alert
        time.sleep(3)
        alert = self.browser.switch_to_alert()
        alert.dismiss()

    def parse_date(self, d):
        return d['mm'] + '/' + d['dd'] + '/' + d['yyyy'][-2:]

    def scrape(self):
        if config.dme_scrape_config['link_objects']['link_id']:
            for link_name in config.dme_scrape_config['link_objects']['link_id']:
                file_path = self.download_zip_file(link_name)
                self.extract_merge_save_csv(file_path)
        else:
            file_path = self.download_zip_file()
            self.extract_merge_save_csv(file_path)

    def create_merged_df_dict(self, file_names):
        d = {}
        for file_name in file_names:
            link = file_name.split('_')[4]
            if link not in d:
                d[link] = {'SOURCE': pd.DataFrame(), 'SINK': pd.DataFrame(),
                           'coverage':
                               {'SOURCE': {'15m': 0, 'dailey': 0},
                                'SINK': {'15m': 0, 'dailey': 0}
                                }
                           }

        if not d:
            raise Exception("no files in zip")

        return d

    def get_coverage(self, coverage, add_df):
        coverage['dailey'] += sum(add_df.Interval == 24)
        coverage['15m'] += sum(add_df.Interval == 15)

    def extract_merge_save_csv(self, file_path):
        print("extracting files...")
        zip_file_object = zipfile.ZipFile(file_path, 'r')
        print("preprocessing...")
        merged_df_dict = self.create_merged_df_dict(zip_file_object.namelist())
        for file_name in zip_file_object.namelist():
            link_type = file_name.split('_')[3]
            link = file_name.split('_')[4]
            bytes = zip_file_object.open(file_name).read()
            add_df = pd.read_csv(io.StringIO(bytes.decode('utf-8')), sep=',')
            self.get_coverage(merged_df_dict[link]['coverage'][link_type], add_df)
            merged_df_dict[link][link_type] = merged_df_dict[link][link_type].append(add_df)

        for link in merged_df_dict:
            for link_type in ['SOURCE', 'SINK']:
                daily_coverage = merged_df_dict[link]["coverage"][link_type]["dailey"] / self.covrage
                _15_minutes_coverage = merged_df_dict[link]["coverage"][link_type]["15m"] / (self.covrage * 4 * 24)
                self.preprocess_df(merged_df_dict[link][link_type]).to_csv(
                    config.dme_scrape_config['path_to_data_files'] +
                    link + f'_{link_type}' +
                    f'_daily-coverage-{daily_coverage}' +
                    f'_15m-coverage-{_15_minutes_coverage}' +
                    '.csv',
                    mode='a', index=False)

                print("file saved to {}\ndaily coverage: {}\n15m coverage: {}".format(link + f'_{link_type}',
                                                                                      daily_coverage,
                                                                                      _15_minutes_coverage))

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
        Select(self.browser.find_element_by_xpath(element_xpath['xpath_select'])).select_by_visible_text(
            select['select'])
        filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])

        if mux == 'date':
            filter.send_keys(select_value['dd'] + select_value['mm'] + select_value['yyyy'])
        elif mux == 'tx_site_longitude' or mux == 'tx_site_latitude' or mux == 'rx_site_longitude' or mux == 'rx_site_latitude':
            filter.send_keys(select_value)
        else:
            raise ValueError("mux type is undefined {}".format(mux))

        if select['select'] == 'In range':
            select_value_range = select['value_range']
            filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter_range'])

            if mux == 'date':
                filter.send_keys(select_value_range['dd'] + select_value_range['mm'] + select_value_range['yyyy'])
            elif mux == 'tx_site_longitude' or mux == 'tx_site_latitude' or mux == 'rx_site_longitude' or mux == 'rx_site_latitude':
                filter.send_keys(select_value_range)
            else:
                raise ValueError("mux type is undefined {}".format(mux))

        self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

    def download_zip_file(self, link_name=None):
        # link id
        if link_name:
            element_xpath = self.xpaths['link_id']
            self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
            filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])
            filter.send_keys(link_name)
            self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

        # date
        self.ranged_filter('date')

        # tx site longitude
        self.ranged_filter('tx_site_longitude')

        # tx site latitude
        self.ranged_filter('tx_site_latitude')

        # rx site longitude
        self.ranged_filter('rx_site_longitude')

        # rx site latitude
        self.ranged_filter('rx_site_latitude')

        # download
        print('starting download...')
        number_of_files = len([file for file in os.listdir(self.root_download) if file not in self.exclude_file])
        self.browser.find_element_by_xpath(self.xpaths['xpath_download']).click()
        th = threading.Thread(target=self.background_task, args=(number_of_files,), kwargs={'delta': 1})
        th.start()

        # wait here for the result of thread
        th.join()
        time.sleep(15)

        file_path = max([self.root_download + f for f in os.listdir(self.root_download) if f not in self.exclude_file],
                        key=os.path.getctime)

        print('download of CMK-link: {} complete\nfile path: {}'.format(link_name, file_path))
        return file_path

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
            if 'cldb' in file:
                try:
                    os.remove(self.root_download + file)
                except PermissionError:
                    shutil.rmtree(self.root_download + file)
                except Exception:
                    raise Exception('unable to remove file : {}'.format(self.root_download + file))


DME_Scrapper_obj().scrape()
