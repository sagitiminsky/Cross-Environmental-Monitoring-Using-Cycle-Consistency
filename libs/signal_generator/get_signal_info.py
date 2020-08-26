import config as config
from libs.links.link_object.link_obj import LinkObj
import numpy as np
from itertools import cycle


class GetSignalInfo:
    def __init__(self, mock=None):
        self.link_names = config.signal_names
        self.links = {}
        self.ticks = 0
        for link_name in self.link_names:
            self.links[link_name] = {
                'cycle': cycle(np.linspace(0, 2 * np.pi, config.max_window_size['15m'])),
                'link_obj': LinkObj(link_name=link_name, mock=mock, debug=True)}

    def measure(self,time_scale=None):
        self.ticks = self.ticks + 1

        if self.ticks >= config.time_scale_to_15m['3mo']:
            self.ticks = 0

        for link_name in self.link_names:
            value = self.links[link_name]['cycle'].__next__()

            link_object = self.links[link_name]['link_obj']
            link_object.enqueue({'value': value})
