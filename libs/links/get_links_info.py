from libs.links.link_object.link_obj import LinkObj
import requests
from bs4 import BeautifulSoup
import apps.ai.config as config


class GetLinksInfo:
    def __init__(self, mock=None):
        self.link_names = config.link_names
        self.links = {}
        self.ticks = 0
        for link_name in self.link_names:
            self.links[link_name] = {'link': f'https://finance.yahoo.com/quote/{stock_name}?p=',
                                       'stock_obj': LinkObj(stock_name=link_name, mock=mock, sin=False)}

    def measure_sch(self):

        self.ticks = self.ticks + 1

        if self.ticks >= config.time_scale2seconds['3mo']:
            self.ticks = 0

        for link_name in self.links:
            value, volume = self.get_cur_price(link_name)

            stock_object = self.links[link_name]['stock_obj']
            stock_object.enqueue({'value': value, 'volume': volume})

    def measure(self,mock=None):

        if mock == None:
            self.measure_sch()

        else:  # unittest
            for link_name in self.links:
                stock_object = self.links[link_name]['stock_obj']
                stock_object.enqueue(mock)

            return True

    def get_cur_price(self, stock_name, mock=None):
        if mock == None:
            tries = 100
            for n in range(tries):
                try:
                    r = requests.get(f'https://finance.yahoo.com/quote/{stock_name}?p=')
                    soup = BeautifulSoup(r.text, "lxml")
                    return float(
                        soup.find_all('div', {'class': 'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text), \
                           int(soup.find_all('td', {'class': "Ta(end) Fw(600) Lh(14px)"})[6].find('span').text.replace(
                               ',', ''))
                except ConnectionError:
                    print(f"Connection Dropped, retry number: {n}")

            raise ConnectionError(f"Connection Lost - tried {tries} times - bye bye ")

        else:
            return mock
