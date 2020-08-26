from libs.links.queues.queue.queue import Queue
from libs.links.queues.candle.candle import Candle
import config as config
import numpy as np


class LinkObj:
    '''
    Thread-safe, memory-efficient, maximally-sized queue supporting queueing and
    dequeueing in worst-case O(1) time.
    '''

    def __init__(self, link_name, mock=None, debug=False):
        '''
        Initialize this queue to the empty queue.

        Parameters
        ----------
        max_size : int
            Maximum number of items contained in this queue. Defaults to 10.
        '''

        self._15m_counter = 0
        if not debug:

            # graphs - scraped
            self.graphs_obj = Graphs_Obj(link_name=link_name, mock=mock)

            self.time_scales = {
                # initialize 1s with 1m closing values
                '15m': Queue(init_list=self.graphs_obj.graphs['1m']['close'], maxlen=config.max_window_size['1s']),
                '30m': Candle(init_dict=self.graphs_obj.graphs['30m'], time_scale='30m'),
                '1h': Candle(init_dict=self.graphs_obj.graphs['1h'], time_scale='1h'),
                '1d': Candle(init_dict=self.graphs_obj.graphs['1d'], time_scale='1d'),
                '5d': Candle(init_dict=self.graphs_obj.graphs['5d'], time_scale='5d'),
                '1mo': Candle(init_dict=self.graphs_obj.graphs['1mo'], time_scale='1mo'),
                '3mo': Candle(init_dict=self.graphs_obj.graphs['3mo'], time_scale='3mo')
            }

        else:

            self.time_scales = {
                # initialize 1s with 1m closing values
                '15m': Queue(init_list=self.apply_trig_func(func=link_name,
                                                            arr=np.linspace(0, 2 * np.pi, config.max_window_size[
                                                                '15m'])), maxlen=config.max_window_size['15m']),

                '30m': Candle(init_dict=self.sin_wave_candle_generator(signal_name=link_name, time_scale='30m'),
                              time_scale='30m'),
                '1h': Candle(init_dict=self.sin_wave_candle_generator(signal_name=link_name, time_scale='1h'),
                             time_scale='1h'),
                '1d': Candle(init_dict=self.sin_wave_candle_generator(signal_name=link_name, time_scale='1d'),
                             time_scale='1d'),
                '1w': Candle(init_dict=self.sin_wave_candle_generator(signal_name=link_name, time_scale='1w'),
                             time_scale='1w'),
                '1mo': Candle(init_dict=self.sin_wave_candle_generator(signal_name=link_name, time_scale='1mo'),
                              time_scale='1mo'),
                '3mo': Candle(init_dict=self.sin_wave_candle_generator(signal_name=link_name, time_scale='3mo'),
                              time_scale='3mo')
            }

    def __str__(self):
        return str(self.time_scales['1s']._queue)

    def sin_wave_candle_generator(self, signal_name, time_scale):
        return {
            'open': self.apply_trig_func(func=signal_name, arr=np.linspace(0, 2 * np.pi,
                                                                           config.max_window_size[time_scale]) + 1),
            'close': self.apply_trig_func(func=signal_name, arr=np.linspace(0, 2 * np.pi,
                                                                            config.max_window_size[time_scale]) - 1),
            'high': self.apply_trig_func(func=signal_name, arr=np.linspace(0, 2 * np.pi,
                                                                           config.max_window_size[time_scale]) + 2),
            'low': self.apply_trig_func(func=signal_name, arr=np.linspace(0, 2 * np.pi,
                                                                          config.max_window_size[time_scale]) - 2),
            'volume': np.zeros(config.max_window_size[time_scale])}

    def apply_trig_func(self, func, arr):
        if func == 'sin':
            return np.sin(arr)
        elif func == 'cos':
            return np.cos(arr)
        else:
            raise Exception("function not supported in signal generator")

    def enqueue(self, item):
        '''
        Queues the passed item (i.e., pushes this item onto the tail of this
        queue).

        If this queue is already full, the item at the head of this queue
        is silently removed from this queue *before* the passed item is
        queued.
        '''

        self.time_scales['15m'].enqueue(item['value'])
        self._15m_counter = self._15m_counter + 1

        if self._15m_counter % config.time_scale_to_15m['30m'] == 0:
            self.time_scales['30m'].enqueue(self.insert(live_by='15m', interval2live_by=2))

        if self._15m_counter % config.time_scale_to_15m['1h'] == 0: self.time_scales['1h'].enqueue(
            self.insert(live_by='30m',interval2live_by=2))

        if self._15m_counter % config.time_scale_to_15m['1d'] == 0: self.time_scales['1d'].enqueue(
            self.insert(live_by='1h',interval2live_by=24))

        if self._15m_counter % config.time_scale_to_15m['1w'] == 0: self.time_scales['1w'].enqueue(
            self.insert(live_by='1d',interval2live_by=7))

        if self._15m_counter % config.time_scale_to_15m['1mo'] == 0: self.time_scales['1mo'].enqueue(
            self.insert(live_by='1w',interval2live_by=4))

        if self._15m_counter % config.time_scale_to_15m['3mo'] == 0: self.time_scales['3mo'].enqueue(
            self.insert(live_by='1mo',interval2live_by=3))

        # initialize sec_counter to zero
        if self._15m_counter >= config.time_scale_to_15m['3mo']:
            self._15m_counter = 0

    def insert(self, live_by, interval2live_by):

        live_by_Queue = self.time_scales[live_by]

        if live_by == '15m':  # Queue
            return {'open': list(live_by_Queue._queue)[-interval2live_by],
                    'low': min(list(live_by_Queue._queue)[-interval2live_by:]),
                    'high': max(list(live_by_Queue._queue)[-interval2live_by:]),
                    'close': list(live_by_Queue._queue)[-1]}

        else:  # Candle
            return {'open': list(live_by_Queue.candle['open']._queue)[-interval2live_by],
                    'low': min(list(live_by_Queue.candle['low']._queue)[-interval2live_by:]),
                    'high': max(list(live_by_Queue.candle['high']._queue)[-interval2live_by:]),
                    'close': list(live_by_Queue.candle['close']._queue)[-1]}
