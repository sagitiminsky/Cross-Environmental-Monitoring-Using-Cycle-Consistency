import time
import config as config
import os
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

class Scrapper_obj:

    def __init__(self, mock=None):
        self.chrome_options = Options()
        # self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument("--disable-popup-blocking")
        self.delay = 30
        self.selector = '//*[@id="btnExportByFilter"]'
        self.scrape_config = config.scrape_config
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
            }
        }
        self.root_download = '/Users/sagit/Downloads/'
        self.root_data_files=self.scrape_config['path_to_data_files']
        self.exclude_file = ['.DS_Store', '.localized', '.com.google.Chrome.tlwaEL']
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=self.chrome_options)

        # clear old files from downloads
        self.delete_prev_from_downloads_if_poss()
        self.delete_prev_data_files_if_poss()

        # log in
        self.browser.get(self.scrape_config['url'])
        self.log_in(self.browser)

        # accept alert
        time.sleep(3)
        alert = self.browser.switch_to_alert()
        alert.dismiss()

    def scrape(self):
        for link_name in self.scrape_config['link_objects']:
            file = self.download_zip_file(link_name)
            self.extract_merge_save_csv(file,link_name)

    def extract_merge_save_csv(self,file_path,link_name):
        zip_file_object = zipfile.ZipFile(file_path, 'r')
        merged_df = pd.DataFrame()
        for file_name in zip_file_object.namelist():
            bytes = zip_file_object.open(file_name).read()
            merged_df=self.preprocess_df(merged_df.append(pd.read_csv(io.StringIO(bytes.decode('utf-8')),sep=',')))

        merged_df.to_csv(self.root_data_files+link_name+'.csv',mode='a',index=False)

    def preprocess_df(self,df):
        return df

    def get_link_config(self, link_name):
        return {'link_name': link_name}

    def background_task(self, prev_number_of_files, delta):
        if len(os.listdir(self.root_download)) - prev_number_of_files == delta:
            return

    def download_zip_file(self, link_name):
        # ready to download
        print('starting download...')

        link_obj = self.scrape_config['link_objects'][link_name]

        # link id
        element_xpath = self.xpaths['link_id']
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])
        filter.send_keys(link_name)
        self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

        # date
        date = link_obj['date']
        date_value = date['value']
        element_xpath = self.xpaths['date']
        self.browser.find_element_by_xpath(element_xpath['xpath_open']).click()
        Select(self.browser.find_element_by_xpath(element_xpath['xpath_select'])).select_by_visible_text(date['select'])
        filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter'])
        filter.send_keys(date_value['dd'] + date_value['mm'] + date_value['yyyy'])

        if date['select'] == 'In range':
            filter = self.browser.find_element_by_xpath(element_xpath['xpath_filter_range'])
            filter.send_keys(date_value['dd'] + date_value['mm'] + date_value['yyyy'])

        self.browser.find_element_by_xpath(element_xpath['xpath_apply']).click()

        # download
        number_of_files = len([file for file in os.listdir(self.root_download) if file not in self.exclude_file])
        self.browser.find_element_by_xpath(self.xpaths['xpath_download']).click()
        th = threading.Thread(target=self.background_task, args=(number_of_files,), kwargs={'delta': 1})
        th.start()

        # wait here for the result of thread
        th.join()
        time.sleep(3)


        file_path = max([self.root_download + f for f in os.listdir(self.root_download) if f not in self.exclude_file],key=os.path.getctime)

        print('download of CMK-link: {} complete\nfile path: {}'.format(link_name, file_path))
        return file_path

    def log_in(self, browser):
        remember_me_xpth = '/html/body/div/form/div[4]/label/input'
        submit_button = '/html/body/div/form/div[3]/button'

        username = browser.find_element_by_name("username")
        password = browser.find_element_by_name("password")

        username.send_keys(self.scrape_config['username'])
        password.send_keys(self.scrape_config['password'])

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
            except:
                pass

    def delete_prev_from_downloads_if_poss(self):
        for file in os.listdir(self.root_download):
            if 'cldb' in file:
                try:
                    os.remove(self.root_download + file)
                except FileNotFoundError:
                    raise FileNotFoundError('unable to remove file : {}'.format(self.root_download + file))


Scrapper_obj().scrape()
