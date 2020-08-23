from libs.signal_generator.get_signal_info import GetSignalInfo
from libs.links.get_links_info import GetLinksInfo
from libs.dataset.models.perceptron.dataset import DateSet
from libs.dl_models.dl_models import DLModels
from libs.callback.callback import CallBack
from libs.threading.threading import CustomTimer
import config as config
from tqdm import tqdm
from threading import Thread


if __name__=='main':

    prediction_type = config.prediction_type
    window_size = config.window_size
    link_names = config.signal_names if config.TEST else config.link_names

    links_obj = GetSignalInfo() if config.TEST else GetLinksInfo()
    dl_models_obj = DLModels(config.prediction_type, callback=CallBack())

    for i in tqdm(range(config.window_size + 10 ** 3)):
        m = CustomTimer(1.0, links_obj.measure)
        m.start()
        m.join()

        for time_scale in config.time_scales:
            trainX, trainY, testX, testY = DateSet(link_names).dataset_handler(time_scale, links_obj, dl_models_obj)

            # todo: this only works for MANY2ONE becuse of config.signal_name[0]
            link_monitor = links_obj.links[links_obj[0]]['stock_obj'].time_scales[time_scale]

            dl_models_obj.fit(time_scale, trainX, trainY, testX, testY)

            if links_obj.ticks % config.time_scale_to_15m[time_scale] == 0:
                thread = Thread(target=dl_models_obj.send2wandb(time_scale, trainX, trainY, testX, testY, link_monitor))
                thread.start()
                thread.join()
