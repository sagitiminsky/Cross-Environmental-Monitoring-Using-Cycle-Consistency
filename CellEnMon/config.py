PYTHONPATH="/Users/sagit/Desktop/CellEnMon-Research"
from datetime import datetime as dt
from datetime import timedelta as dt_delta
import os
import torch
import numpy as np

MAC = False
download_path = '/Users/sagitiminsky/Downloads' if MAC == True else '/home/sagit/Downloads'

bucket_creds = "CellEnMon/cellenmon-e840a9ba53e8.json"


def parse_date(d):
    return d['mm'] + '/' + d['dd'] + '/' + d['yyyy'][-2:]


def create_directory_if_does_not_exist(outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)


def add_days_to_date(date, delta_days=0):
    res = dt.strptime(parse_date(date), "%m/%d/%y") + dt_delta(days=delta_days)
    return {
        'datetime_rep': res,
        'str_rep': f"{str(res.month).zfill(2)}/{str(res.day).zfill(2)}/{res.year}",
        'str_rep_with_replace': f"{str(res.month).zfill(2)}{str(res.day).zfill(2)}{res.year}",
        'day_str_rep': f"{str(res.day).zfill(2)}",
        'month_str_rep': f"{str(res.month).zfill(2)}",
        'year_str_rep': f"{res.year}",
        'str_rep_ddmmyyyy': f"{str(res.day).zfill(2)}{str(res.month).zfill(2)}{res.year}",
        'str_rep_mmddyyyy': f"{str(res.month).zfill(2)}{str(res.day).zfill(2)}{res.year}"
    }

#09062011_13082011 dutch
#01012013_01022013 israel
smoothing_dme=1
smoothing_ims="xx" # not used yet

TRAIN_RADIUS=30

# 10KM <-> 2 pairs
# 30KM <-> 9 pairs

def func_fit(x, a):
    x=torch.from_numpy(np.array(x))
    b=torch.from_numpy(np.array(a))
    return a*x


VALIDATION_RADIUS=100

export_type="israel"
date = {
    'value': {
        'dd': '01',
        'mm': '01',
        'yyyy': '2015'
    },
    'value_range': {
        'dd': '01',
        'mm': '02',
        'yyyy': '2015'
    }
}

date_str_rep = add_days_to_date(date['value'])['str_rep_with_replace'] + '_' + add_days_to_date(date['value_range'])[
    'str_rep_with_replace']
date_datetime_rep = add_days_to_date(date['value'])['datetime_rep']
start_date_str_rep_mmddyyyy = add_days_to_date(date['value'])['str_rep_mmddyyyy']
start_date_str_rep_ddmmyyyy = add_days_to_date(date['value'])['str_rep_ddmmyyyy']

end_date_str_rep_mmddyyyy = add_days_to_date(date['value_range'])['str_rep_mmddyyyy']
end_date_str_rep_ddmmyyyy = add_days_to_date(date['value_range'])['str_rep_ddmmyyyy']

months = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

#########################################
############## RADAR SCRAPPER ###########
#########################################

radar_root_files = f"datasets/radar/raw/{add_days_to_date(date['value'])['str_rep_with_replace']}_{add_days_to_date(date['value_range'])['str_rep_with_replace']}/"  # MM/DD/YYYY
radar_root_values = 'datasets/radar/processed'

#######################################
############## IMS SCRAPPER ###########
#######################################

ims_root_files = f"CellEnMon/datasets/ims/{start_date_str_rep_ddmmyyyy}_{end_date_str_rep_ddmmyyyy}"  # DD/MM/YYYY
ims_mapping = [
    {
        "stationId": 2,
        "name": "AVNE ETAN",
        "shortName": "AVNE ETA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.817,
            "longitude": 35.763
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 17,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 6,
        "name": "BET ZAYDA",
        "shortName": "BET ZAYD",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.881,
            "longitude": 35.653
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 14,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 8,
        "name": "ZEMAH",
        "shortName": "ZEMAH",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.704,
            "longitude": 35.584
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 5,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 6,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 7,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 8,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 9,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 10,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 11,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 14,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 17,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 20,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 21,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 10,
        "name": "MEROM GOLAN PICMAN",
        "shortName": "MEROM GO",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.133,
            "longitude": 35.783
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 11,
        "name": "YAVNEEL",
        "shortName": "YAVNEEL",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.683,
            "longitude": 35.516
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 18,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 20,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 21,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 13,
        "name": "TAVOR KADOORIE",
        "shortName": "TAVOR KA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.7,
            "longitude": 35.4
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 18,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 20,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 16,
        "name": "AFULA NIR HAEMEQ",
        "shortName": "AFULA NI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.597,
            "longitude": 35.278
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 17,
        "name": "TEL YOSEF 20060907",
        "shortName": "TEL YOSE",
        "stationsTag": "(None)",
        "location": {
            "latitude": None,
            "longitude": None
        },
        "timebase": 10,
        "active": False,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": False,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 18,
        "name": "EDEN FARM 20080706",
        "shortName": "EDEN FAR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.47,
            "longitude": 35.489
        },
        "timebase": 10,
        "active": False,
        "owner": "ims",
        "regionId": 6,
        "monitors": [
            {
                "channelId": 1,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSV",
                "alias": None,
                "active": False,
                "typeId": 19,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WS3.5m",
                "alias": None,
                "active": False,
                "typeId": 9,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 6,
                "name": "Licor",
                "alias": None,
                "active": False,
                "typeId": 150,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 7,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD10m",
                "alias": None,
                "active": False,
                "typeId": 44,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 11,
                "name": "T.5m",
                "alias": None,
                "active": False,
                "typeId": 54,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "TG",
                "alias": None,
                "active": False,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 13,
                "name": "T-.1m",
                "alias": None,
                "active": False,
                "typeId": 48,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 14,
                "name": "T-.2m",
                "alias": None,
                "active": False,
                "typeId": 49,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 15,
                "name": "T-.5m",
                "alias": None,
                "active": False,
                "typeId": 37,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Vbatt",
                "alias": None,
                "active": False,
                "typeId": 2,
                "pollutantId": None,
                "units": "volt",
                "description": None
            },
            {
                "channelId": 18,
                "name": "TT",
                "alias": None,
                "active": False,
                "typeId": 248,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 19,
                "name": "TW",
                "alias": None,
                "active": True,
                "typeId": 31,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 20,
                "name": "EVAP",
                "alias": None,
                "active": False,
                "typeId": 62,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 21,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 22,
                "name": "WS3.5m",
                "alias": None,
                "active": False,
                "typeId": 9,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 23,
                "name": "STAB",
                "alias": None,
                "active": False,
                "typeId": 138,
                "pollutantId": None,
                "units": "_",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 20,
        "name": "QARNE SHOMERON",
        "shortName": "QARNE SH",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.166,
            "longitude": 35.099
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 21,
        "name": "ARIEL",
        "shortName": "ARIEL",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.106,
            "longitude": 35.212
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 22,
        "name": "JERUSALEM GIVAT RAM",
        "shortName": "JERUSALE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.771,
            "longitude": 35.197
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 12,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 17,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 20,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 23,
        "name": "JERUSALEM CENTRE",
        "shortName": "JERUSALE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.781,
            "longitude": 35.222
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 17,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            },
            {
                "channelId": 18,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 24,
        "name": "HAR HARASHA",
        "shortName": "HAR HARA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.933,
            "longitude": 35.15
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 25,
        "name": "NETIV HALAMED HE",
        "shortName": "NETIV HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.683,
            "longitude": 34.983
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 13,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 26,
        "name": "HAIFA PORT",
        "shortName": "HAIFA PO",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.822,
            "longitude": 34.999
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 6,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "BP",
                "alias": None,
                "active": False,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Vbatt",
                "alias": None,
                "active": False,
                "typeId": 2,
                "pollutantId": None,
                "units": "volt",
                "description": None
            },
            {
                "channelId": 13,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 16,
                "name": "STAB",
                "alias": None,
                "active": False,
                "typeId": 138,
                "pollutantId": None,
                "units": "_",
                "description": None
            },
            {
                "channelId": 17,
                "name": "TW",
                "alias": None,
                "active": True,
                "typeId": 31,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 28,
        "name": "SHANI",
        "shortName": "SHANI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.35,
            "longitude": 35.049
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 15,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 29,
        "name": "ARAD",
        "shortName": "ARAD",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.251,
            "longitude": 35.186
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 13,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 14,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 15,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 30,
        "name": "GILGAL",
        "shortName": "GILGAL",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.983,
            "longitude": 35.45
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 16,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 20,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 32,
        "name": "NEWE ZOHAR UNI",
        "shortName": "NEWE ZOH",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.152,
            "longitude": 35.365
        },
        "timebase": 10,
        "active": True,
        "owner": "None",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 4,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 16,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 33,
        "name": "HAZEVA",
        "shortName": "HAZEVA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.51,
            "longitude": 35.14
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 5,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 15,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 16,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 17,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 35,
        "name": "PARAN 20060124",
        "shortName": "PARAN 20",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.3696,
            "longitude": 35.15
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": False,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 36,
        "name": "YOTVATA",
        "shortName": "YOTVATA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 29.886,
            "longitude": 35.078
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 5,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 14,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 18,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 19,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 20,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 21,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 24,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 41,
        "name": "HAIFA REFINERIES",
        "shortName": "HAIFA RE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.799,
            "longitude": 35.049
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 42,
        "name": "HAIFA UNIVERSITY",
        "shortName": "HAIFA UN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.762,
            "longitude": 35.021
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 12,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 43,
        "name": "HAIFA TECHNION",
        "shortName": "HAIFA TE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.766,
            "longitude": 35.016
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 5,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 11,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 12,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 13,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 20,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 44,
        "name": "EN KARMEL",
        "shortName": "EN KARME",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.682,
            "longitude": 34.96
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 16,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 45,
        "name": "ZIKHRON YAAQOV",
        "shortName": "ZIKHRON",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.566,
            "longitude": 34.95
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 13,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 46,
        "name": "HADERA PORT",
        "shortName": "HADERA P",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.474,
            "longitude": 34.883
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 5,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 18,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 20,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 54,
        "name": "BET DAGAN",
        "shortName": "BET DAGA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.009,
            "longitude": 34.814
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 13,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 15,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 18,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 26,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 58,
        "name": "BESOR FARM",
        "shortName": "BESOR FA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.273,
            "longitude": 34.389
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 15,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 16,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 17,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 59,
        "name": "BEER SHEVA",
        "shortName": "BEER SHE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.333,
            "longitude": 34.783
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 16,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 19,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 60,
        "name": "BEER SHEVA UNI",
        "shortName": "BEER SHE",
        "stationsTag": "(None)",
        "location": {
            "latitude": None,
            "longitude": None
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 2,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 3,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TD",
                "alias": None,
                "active": False,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 62,
        "name": "ZEFAT HAR KENAAN",
        "shortName": "ZEFAT HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.982,
            "longitude": 35.507
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 16,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 64,
        "name": "ELAT",
        "shortName": "ELAT",
        "stationsTag": "(None)",
        "location": {
            "latitude": 29.553,
            "longitude": 34.954
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 5,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 7,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 13,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 14,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 21,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 22,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 23,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 65,
        "name": "SEDOM",
        "shortName": "SEDOM",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.032,
            "longitude": 35.391
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 2,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 3,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 4,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 8,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 9,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 16,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 18,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 19,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 20,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 21,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 22,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 25,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 67,
        "name": "EN HASHOFET",
        "shortName": "EN HASHO",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.633,
            "longitude": 35.099
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 69,
        "name": "MIZPE RAMON 20080514",
        "shortName": "MIZPE RA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.614,
            "longitude": 34.797
        },
        "timebase": 10,
        "active": False,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 12,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 14,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 73,
        "name": "ELON",
        "shortName": "ELON",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.066,
            "longitude": 35.217
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "TW",
                "alias": None,
                "active": True,
                "typeId": 31,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 17,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 74,
        "name": "QEVUZAT YAVNE",
        "shortName": "QEVUZAT",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.816,
            "longitude": 34.7
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": False,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 75,
        "name": "BEIT JIMAL",
        "shortName": "BEIT JIM",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.716,
            "longitude": 34.966
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 77,
        "name": "ROSH ZURIM",
        "shortName": "ROSH ZUR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.665,
            "longitude": 35.124
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 78,
        "name": "AFEQ",
        "shortName": "AFEQ",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.848,
            "longitude": 35.112
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 79,
        "name": "DOROT",
        "shortName": "DOROT",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.5,
            "longitude": 34.633
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 16,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 17,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 82,
        "name": "NEGBA",
        "shortName": "NEGBA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.658,
            "longitude": 34.681
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 15,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 17,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 85,
        "name": "BET DAGAN RAD",
        "shortName": "BET DAGA",
        "stationsTag": "(None)",
        "location": {
            "latitude": None,
            "longitude": None
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 13,
        "monitors": [
            {
                "channelId": 6,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 7,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 8,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 90,
        "name": "ITAMAR",
        "shortName": "ITAMAR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.166,
            "longitude": 35.349
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 98,
        "name": "SEDE BOQER",
        "shortName": "SEDE BOQ",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.871,
            "longitude": 34.796
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 99,
        "name": "DEIR HANNA",
        "shortName": "DEIR HAN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.863,
            "longitude": 35.374
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 106,
        "name": "ROSH HANIQRA",
        "shortName": "ROSH HAN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.081,
            "longitude": 35.107
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 107,
        "name": "EN HAHORESH",
        "shortName": "EN HAHOR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.388,
            "longitude": 34.938
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 112,
        "name": "ZOMET HANEGEV",
        "shortName": "ZOMET HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.066,
            "longitude": 34.849
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 115,
        "name": "LEV KINERET",
        "shortName": "LEV KINE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.816,
            "longitude": 35.599
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 121,
        "name": "HAFEZ HAYYIM",
        "shortName": "HAFEZ HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.793,
            "longitude": 34.804
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Vbatt",
                "alias": None,
                "active": False,
                "typeId": 2,
                "pollutantId": None,
                "units": "volt",
                "description": None
            },
            {
                "channelId": 16,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 123,
        "name": "AMMIAD",
        "shortName": "AMMIAD",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.916,
            "longitude": 35.533
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 124,
        "name": "ASHDOD PORT",
        "shortName": "ASHDOD P",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.835,
            "longitude": 34.638
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 178,
        "name": "TEL AVIV COAST",
        "shortName": "TEL AVIV",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.059,
            "longitude": 34.759
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 13,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 186,
        "name": "NEWE YAAR",
        "shortName": "NEWE YAA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.709,
            "longitude": 35.179
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 12,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 188,
        "name": "ZOVA",
        "shortName": "ZOVA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.47,
            "longitude": 35.07
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 202,
        "name": "KEFAR BLUM",
        "shortName": "KEFAR BL",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.173,
            "longitude": 35.613
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 14,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 205,
        "name": "ESHHAR",
        "shortName": "ESHHAR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.883,
            "longitude": 35.299
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 206,
        "name": "EDEN FARM",
        "shortName": "EDEN FAR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.47,
            "longitude": 35.489
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 13,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 17,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 18,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 19,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 20,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 207,
        "name": "PARAN",
        "shortName": "PARAN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.367,
            "longitude": 35.148
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 208,
        "name": "ASHQELON PORT",
        "shortName": "ASHQELON",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.633,
            "longitude": 34.516
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 210,
        "name": "METZOKE DRAGOT",
        "shortName": "METZOKE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.583,
            "longitude": 35.383
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 211,
        "name": "EN GEDI",
        "shortName": "EN GEDI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.419,
            "longitude": 35.386
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 212,
        "name": "BET DAGAN_1m",
        "shortName": "BET DAGA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.0073,
            "longitude": 34.8138
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 13,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 3,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 218,
        "name": "MAALE ADUMMIM",
        "shortName": "MAALE AD",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.773,
            "longitude": 35.296
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 224,
        "name": "MAALE GILBOA",
        "shortName": "MAALE GI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.481,
            "longitude": 35.415
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 227,
        "name": "GAMLA",
        "shortName": "GAMLA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.907,
            "longitude": 35.748
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 228,
        "name": "BET HAARAVA",
        "shortName": "BET HAAR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.805,
            "longitude": 35.501
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 232,
        "name": "NEOT SMADAR",
        "shortName": "NEOT SMA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.048,
            "longitude": 35.023
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 233,
        "name": "KEFAR NAHUM",
        "shortName": "KEFAR NA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.883,
            "longitude": 35.579
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 236,
        "name": "GAT",
        "shortName": "GAT",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.616,
            "longitude": 34.783
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 238,
        "name": "ZEFAT HAR KENAAN_1m",
        "shortName": "ZEFAT HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.98,
            "longitude": 35.507
        },
        "timebase": 1,
        "active": False,
        "owner": "Rain_1m",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": False,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 239,
        "name": "ELON_1m",
        "shortName": "ELON_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.0653,
            "longitude": 35.2173
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 240,
        "name": "ARAD_1m",
        "shortName": "ARAD_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.25,
            "longitude": 35.1855
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 241,
        "name": "KEFAR GILADI",
        "shortName": "KEFAR GI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.241,
            "longitude": 35.57
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 242,
        "name": "ARIEL_1m",
        "shortName": "ARIEL_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.1056,
            "longitude": 35.2114
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 243,
        "name": "ASHDOD PORT_1m",
        "shortName": "ASHDOD P",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.8342,
            "longitude": 34.6377
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 6,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 244,
        "name": "BESOR FARM_1m",
        "shortName": "BESOR FA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.2716,
            "longitude": 34.3894
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 245,
        "name": "DOROT_1m",
        "shortName": "DOROT_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.5036,
            "longitude": 34.648
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 246,
        "name": "EN HAHORESH_1m",
        "shortName": "EN HAHOR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.3877,
            "longitude": 34.9376
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 247,
        "name": "QARNE SHOMERON_1m",
        "shortName": "QARNE SH",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.1752,
            "longitude": 35.0959
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 248,
        "name": "JERUSALEM CENTRE_1m",
        "shortName": "JERUSALE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.7806,
            "longitude": 35.2217
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 249,
        "name": "JERUSALEM GIVAT RAM_1m",
        "shortName": "JERUSALE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.7704,
            "longitude": 35.1973
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 250,
        "name": "SEDOM_1m",
        "shortName": "SEDOM_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.0306,
            "longitude": 35.3919
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 251,
        "name": "SEDE BOQER_1m",
        "shortName": "SEDE BOQ",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.8702,
            "longitude": 34.795
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 252,
        "name": "NEGBA_1m",
        "shortName": "NEGBA_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.6585,
            "longitude": 34.6798
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 259,
        "name": "NAHSHON",
        "shortName": "NAHSHON",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.834,
            "longitude": 34.961
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 16,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 263,
        "name": "GALED",
        "shortName": "GALED",
        "stationsTag": "(None)",
        "location": {
            "latitude": None,
            "longitude": None
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 265,
        "name": "MIZPE RAMON 20120927",
        "shortName": "MIZPE RA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.614,
            "longitude": 34.797
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 269,
        "name": "HARASHIM",
        "shortName": "HARASHIM",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.956,
            "longitude": 35.328
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": False,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": False,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": False,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": False,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": False,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": False,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": False,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": False,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 270,
        "name": "SHAARE TIQWA 20161205",
        "shortName": "SHAARE T",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.127,
            "longitude": 35.026
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 271,
        "name": "AVDAT",
        "shortName": "AVDAT",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.8,
            "longitude": 34.77
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 274,
        "name": "NIZZAN",
        "shortName": "NIZZAN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.733,
            "longitude": 34.635
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 275,
        "name": "HAKFAR HAYAROK",
        "shortName": "HAKFAR H",
        "stationsTag": "(None)",
        "location": {
            "latitude": None,
            "longitude": None
        },
        "timebase": 5,
        "active": True,
        "owner": "iec",
        "regionId": 13,
        "monitors": [
            {
                "channelId": 9,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 12,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 13,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 276,
        "name": "GILGAL_1m",
        "shortName": "GILGAL_1",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.9973,
            "longitude": 35.4509
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 277,
        "name": "ITAMAR_1m",
        "shortName": "ITAMAR_1",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.1598,
            "longitude": 35.3513
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 278,
        "name": "HAR HARASHA_1m",
        "shortName": "HAR HARA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.9449,
            "longitude": 35.1499
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 279,
        "name": "BET HAARAVA_1m",
        "shortName": "BET HAAR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.8052,
            "longitude": 35.501
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 280,
        "name": "MAALE ADUMMIM_1m",
        "shortName": "MAALE AD",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.774,
            "longitude": 35.2961
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 281,
        "name": "ZOVA_1m",
        "shortName": "ZOVA_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.7878,
            "longitude": 35.1258
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 282,
        "name": "NAHSHON_1m",
        "shortName": "NAHSHON_",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.8341,
            "longitude": 34.9616
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 283,
        "name": "HAFEZ HAYYIM_1m",
        "shortName": "HAFEZ HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.791,
            "longitude": 34.805
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 284,
        "name": "QEVUZAT YAVNE_1m",
        "shortName": "QEVUZAT",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.8166,
            "longitude": 34.7244
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 285,
        "name": "GAT_1m",
        "shortName": "GAT_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.6303,
            "longitude": 34.7913
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 286,
        "name": "ROSH ZURIM_1m",
        "shortName": "ROSH ZUR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.6644,
            "longitude": 35.1233
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 287,
        "name": "NETIV HALAMED HE_1m",
        "shortName": "NETIV HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.6898,
            "longitude": 34.9744
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 288,
        "name": "BEIT JIMAL_1m",
        "shortName": "BEIT JIM",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.7248,
            "longitude": 34.9762
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 289,
        "name": "METZOKE DRAGOT_1m",
        "shortName": "METZOKE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.5881,
            "longitude": 35.3916
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 290,
        "name": "EN GEDI_1m",
        "shortName": "EN GEDI_",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.42,
            "longitude": 35.3871
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 291,
        "name": "ASHQELON PORT_1m",
        "shortName": "ASHQELON",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.6394,
            "longitude": 34.5215
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 292,
        "name": "SHANI_1m",
        "shortName": "SHANI_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.3568,
            "longitude": 35.0662
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 293,
        "name": "BEER SHEVA_1m",
        "shortName": "BEER SHE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.2515,
            "longitude": 34.7995
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 294,
        "name": "ZOMET HANEGEV_1m",
        "shortName": "ZOMET HA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.0708,
            "longitude": 34.8513
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 295,
        "name": "HAZEVA_1m",
        "shortName": "HAZEVA_1",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.7787,
            "longitude": 35.2389
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 296,
        "name": "MIZPE RAMON_1m",
        "shortName": "MIZPE RA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.6101,
            "longitude": 34.8046
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 297,
        "name": "PARAN_1m",
        "shortName": "PARAN_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.3655,
            "longitude": 35.1479
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 298,
        "name": "HARASHIM_1m",
        "shortName": "HARASHIM",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.956,
            "longitude": 35.3287
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 299,
        "name": "TEL AVIV COAST_1m",
        "shortName": "TEL AVIV",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.058,
            "longitude": 34.7588
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 13,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 300,
        "name": "AVNE ETAN_1m",
        "shortName": "AVNE ETA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.8174,
            "longitude": 35.7622
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 301,
        "name": "BET ZAYDA_1m",
        "shortName": "BET ZAYD",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.88,
            "longitude": 35.6504
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 302,
        "name": "ZEMAH_1m",
        "shortName": "ZEMAH_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.7024,
            "longitude": 35.5839
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 303,
        "name": "MEROM GOLAN PICMAN_1m",
        "shortName": "MEROM GO",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.1288,
            "longitude": 35.8045
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 304,
        "name": "YAVNEEL_1m",
        "shortName": "YAVNEEL_",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.6978,
            "longitude": 35.5101
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 305,
        "name": "TAVOR KADOORIE_1m",
        "shortName": "TAVOR KA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.7053,
            "longitude": 35.4069
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 306,
        "name": "AFULA NIR HAEMEQ_1m",
        "shortName": "AFULA NI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.596,
            "longitude": 35.2769
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 307,
        "name": "EDEN FARM_1m",
        "shortName": "EDEN FAR",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.4679,
            "longitude": 35.4888
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 309,
        "name": "YOTVATA_1m",
        "shortName": "YOTVATA_",
        "stationsTag": "(None)",
        "location": {
            "latitude": 29.8851,
            "longitude": 35.0771
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 310,
        "name": "HAIFA REFINERIES_1m",
        "shortName": "HAIFA RE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.8034,
            "longitude": 35.0548
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 311,
        "name": "HAIFA UNIVERSITY_1m",
        "shortName": "HAIFA UN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.7611,
            "longitude": 35.0208
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 312,
        "name": "HAIFA TECHNION_1m",
        "shortName": "HAIFA TE",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.7736,
            "longitude": 35.0223
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 313,
        "name": "EN KARMEL_1m",
        "shortName": "EN KARME",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.6808,
            "longitude": 34.9594
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 314,
        "name": "ZIKHRON YAAQOV_1m",
        "shortName": "ZIKHRON",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.5724,
            "longitude": 34.9546
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 315,
        "name": "HADERA PORT_1m",
        "shortName": "HADERA P",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.4732,
            "longitude": 34.8815
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 316,
        "name": "ELAT_1m",
        "shortName": "ELAT_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 29.5526,
            "longitude": 34.952
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 317,
        "name": "EN HASHOFET_1m",
        "shortName": "EN HASHO",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.6028,
            "longitude": 35.0964
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 318,
        "name": "AFEQ_1m",
        "shortName": "AFEQ_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.8466,
            "longitude": 35.1123
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 13,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 319,
        "name": "DEIR HANNA_1m",
        "shortName": "DEIR HAN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.8621,
            "longitude": 35.3741
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 320,
        "name": "ROSH HANIQRA_1m",
        "shortName": "ROSH HAN",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.0806,
            "longitude": 35.1079
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 322,
        "name": "AMMIAD_1m",
        "shortName": "AMMIAD_1",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.915,
            "longitude": 35.5135
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 323,
        "name": "NEWE YAAR_1m",
        "shortName": "NEWE YAA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.7078,
            "longitude": 35.1784
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 324,
        "name": "KEFAR BLUM_1m",
        "shortName": "KEFAR BL",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.1716,
            "longitude": 35.6133
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 325,
        "name": "ESHHAR_1m",
        "shortName": "ESHHAR_1",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.8844,
            "longitude": 35.3015
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 327,
        "name": "GAMLA_1m",
        "shortName": "GAMLA_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.9052,
            "longitude": 35.7487
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 328,
        "name": "MAALE GILBOA_1m",
        "shortName": "MAALE GI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.481,
            "longitude": 35.415
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 7,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 329,
        "name": "NEOT SMADAR_1m",
        "shortName": "NEOT SMA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.048,
            "longitude": 35.0233
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 10,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 330,
        "name": "KEFAR NAHUM_1m",
        "shortName": "KEFAR NA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.8835,
            "longitude": 35.5792
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 332,
        "name": "GALED_1m",
        "shortName": "GALED_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.5564,
            "longitude": 35.0725
        },
        "timebase": 1,
        "active": False,
        "owner": "Rain_1m",
        "regionId": 11,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 335,
        "name": "AVDAT_1m",
        "shortName": "AVDAT_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.7877,
            "longitude": 34.7712
        },
        "timebase": 1,
        "active": True,
        "owner": "Rain_1m",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 336,
        "name": "NIZZAN_1m",
        "shortName": "NIZZAN_1",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.7319,
            "longitude": 34.6348
        },
        "timebase": 1,
        "active": True,
        "owner": "Rain_1m",
        "regionId": 14,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 338,
        "name": "EZUZ",
        "shortName": "EZUZ",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.791,
            "longitude": 34.472
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 343,
        "name": "SHAVE ZIYYON",
        "shortName": "SHAVE ZI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.98,
            "longitude": 35.09
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 344,
        "name": "EZUZ_1m",
        "shortName": "EZUZ_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.7911,
            "longitude": 34.4715
        },
        "timebase": 1,
        "active": True,
        "owner": "Rain_1m",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 345,
        "name": "SHAVE ZIYYON_1m",
        "shortName": "SHAVE ZI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.9836,
            "longitude": 35.0923
        },
        "timebase": 1,
        "active": True,
        "owner": "Rain_1m",
        "regionId": 15,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 346,
        "name": "KEFAR GILADI_1m",
        "shortName": "KEFAR GI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.2404,
            "longitude": 35.5696
        },
        "timebase": 1,
        "active": True,
        "owner": "Rain_1m",
        "regionId": 8,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 349,
        "name": "NEVATIM",
        "shortName": "NEVATIM",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.205,
            "longitude": 34.922
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 350,
        "name": "LAHAV",
        "shortName": "LAHAV",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.3,
            "longitude": 34.87
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 351,
        "name": "LAHAV_1m",
        "shortName": "LAHAV_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.3812,
            "longitude": 34.8729
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 353,
        "name": "AYYELET HASHAHAR",
        "shortName": "AYYELET",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.0244,
            "longitude": 35.5735
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 354,
        "name": "AYYELEY HASHAHAR_1m",
        "shortName": "AYYELEY",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.0244,
            "longitude": 35.5735
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 355,
        "name": "MASSADA",
        "shortName": "MASSADA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.67,
            "longitude": 35.6
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 366,
        "name": "SEDE ELIYYAHU",
        "shortName": "SEDE ELI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.44,
            "longitude": 35.51
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 3,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 4,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 5,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 6,
                "name": "TG",
                "alias": None,
                "active": True,
                "typeId": 33,
                "pollutantId": None,
                "units": "degC",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 367,
        "name": "SEDE ELIYYAHU_1m",
        "shortName": "SEDE ELI",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.439,
            "longitude": 35.5106
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 370,
        "name": "NEVATIM_1m",
        "shortName": "NEVATIM_",
        "stationsTag": "(None)",
        "location": {
            "latitude": 31.205,
            "longitude": 34.9227
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 373,
        "name": "MASSADA_1m",
        "shortName": "MASSADA_",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.6833,
            "longitude": 35.6008
        },
        "timebase": 1,
        "active": False,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 379,
        "name": "MIZPE RAMON",
        "shortName": "MIZPE RAMON",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.6125,
            "longitude": 34.7967
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Id",
                "alias": None,
                "active": False,
                "typeId": 224,
                "pollutantId": None,
                "units": "_",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 380,
        "name": "TEL YOSEF",
        "shortName": "TEL YOSEF",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.5462,
            "longitude": 35.3945
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Id",
                "alias": None,
                "active": False,
                "typeId": 224,
                "pollutantId": None,
                "units": "_",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 381,
        "name": "ASHALIM",
        "shortName": "ASHALIM",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.98,
            "longitude": 34.7
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": True,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WSmax",
                "alias": None,
                "active": True,
                "typeId": 5,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": True,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": True,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": True,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": True,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": True,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": True,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": True,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": True,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": True,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": True,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": True,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 14,
                "name": "Grad",
                "alias": None,
                "active": True,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 15,
                "name": "DiffR",
                "alias": None,
                "active": True,
                "typeId": 77,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 16,
                "name": "NIP",
                "alias": None,
                "active": True,
                "typeId": 79,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            },
            {
                "channelId": 18,
                "name": "BP",
                "alias": None,
                "active": True,
                "typeId": 84,
                "pollutantId": None,
                "units": "mb",
                "description": None
            },
            {
                "channelId": 30,
                "name": "Id",
                "alias": None,
                "active": False,
                "typeId": 224,
                "pollutantId": None,
                "units": "_",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 443,
        "name": "TEL YOSEF_1m",
        "shortName": "Station 443",
        "stationsTag": "(None)",
        "location": {
            "latitude": 32.5462,
            "longitude": 35.3945
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 480,
        "name": "ASHALIM_1m",
        "shortName": "ASHALIM_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 30.9831,
            "longitude": 34.7078
        },
        "timebase": 1,
        "active": True,
        "owner": "ims",
        "regionId": 12,
        "monitors": [
            {
                "channelId": 1,
                "name": "WS",
                "alias": None,
                "active": False,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 498,
        "name": "DAFNA_1m",
        "shortName": "DAFNA_1m",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.2277,
            "longitude": 35.635
        },
        "timebase": 5,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain_1_min",
                "alias": "mm",
                "active": True,
                "typeId": 325,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "Rain_Corr",
                "alias": None,
                "active": True,
                "typeId": 401,
                "pollutantId": None,
                "units": "mm",
                "description": None
            }
        ],
        "StationTarget": ""
    },
    {
        "stationId": 499,
        "name": "DAFNA",
        "shortName": "DAFNA",
        "stationsTag": "(None)",
        "location": {
            "latitude": 33.2277,
            "longitude": 35.635
        },
        "timebase": 10,
        "active": True,
        "owner": "ims",
        "regionId": 9,
        "monitors": [
            {
                "channelId": 1,
                "name": "Rain",
                "alias": None,
                "active": False,
                "typeId": 1,
                "pollutantId": None,
                "units": "mm",
                "description": None
            },
            {
                "channelId": 2,
                "name": "WS_Max",
                "alias": None,
                "active": False,
                "typeId": 13,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 3,
                "name": "WDmax",
                "alias": None,
                "active": False,
                "typeId": 23,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 4,
                "name": "WS",
                "alias": None,
                "active": False,
                "typeId": 4,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 5,
                "name": "WD",
                "alias": None,
                "active": False,
                "typeId": 22,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 6,
                "name": "STDwd",
                "alias": None,
                "active": False,
                "typeId": 3,
                "pollutantId": None,
                "units": "deg",
                "description": None
            },
            {
                "channelId": 7,
                "name": "TD",
                "alias": None,
                "active": False,
                "typeId": 32,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 8,
                "name": "RH",
                "alias": None,
                "active": False,
                "typeId": 27,
                "pollutantId": None,
                "units": "%",
                "description": None
            },
            {
                "channelId": 9,
                "name": "TDmax",
                "alias": None,
                "active": False,
                "typeId": 41,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 10,
                "name": "TDmin",
                "alias": None,
                "active": False,
                "typeId": 42,
                "pollutantId": None,
                "units": "degC",
                "description": None
            },
            {
                "channelId": 11,
                "name": "WS1mm",
                "alias": None,
                "active": False,
                "typeId": 6,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 12,
                "name": "Ws10mm",
                "alias": None,
                "active": False,
                "typeId": 144,
                "pollutantId": None,
                "units": "m/sec",
                "description": None
            },
            {
                "channelId": 13,
                "name": "Time",
                "alias": None,
                "active": False,
                "typeId": 102,
                "pollutantId": None,
                "units": "hhmm",
                "description": None
            },
            {
                "channelId": 15,
                "name": "Id",
                "alias": None,
                "active": False,
                "typeId": 224,
                "pollutantId": None,
                "units": "_",
                "description": None
            },
            {
                "channelId": 16,
                "name": "Grad",
                "alias": None,
                "active": False,
                "typeId": 52,
                "pollutantId": None,
                "units": "w/m2",
                "description": None
            }
        ],
        "StationTarget": ""
    }
]

ims_scrape_config = {
    '_from': f"{add_days_to_date(date['value'])['str_rep']}",  # MM/DD/YYYY
    '_to': f"{add_days_to_date(date['value_range'])['str_rep']}",  # MM/DD/YYYY , delta_days=2
    'total_number_of_ims_stations': len(ims_mapping)
}

def dme_ims_root_file(db_type):
    return f"CellEnMon/datasets/{db_type}/{start_date_str_rep_ddmmyyyy}_{end_date_str_rep_ddmmyyyy}"  # DD/MM/YYYY
################################
######### DME SCRAPPER #########
################################

xpaths = {
    'xpath_download': '//*[@id="btnExport"]',
    'xpath_metadata_download': '//*[@id="dailies"]/div/div[7]/div/div/div[1]/span[2]',
    'link_id': {
        'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[6]/div/div[1]',
        'xpath_select': '',
        'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[6]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div/input',
        'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[6]/div/div[3]/div/div[2]/div/div/div[2]/button[3]',
        'xpath_reset': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[6]/div/div[3]/div/div[2]/div/div/div[2]/button[2]'

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
        'search_box': '//*[@id="ag-mini-filter"]/input',
        'xpath_hc_radio_sink': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[1]/div[2]/div[2]/div/div/div[1]/label',
        'xpath_hc_radio_source': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[1]/div[2]/div[2]/div/div/div[2]/label',
        'xpath_tn_rfinputpower': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[1]/div[2]/div[2]/div/div/div/label/span',
        'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[2]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'

    },
    'data_precentage': {
        'xpath_open': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[3]/div/div[1]',
        'xpath_select': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[3]/div/div[3]/div/div[2]/div/div/div[1]/select[1]',
        'xpath_filter': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[3]/div/div[3]/div/div[2]/div/div/div[1]/div[1]/div[1]/input',
        'xpath_apply': '//*[@id="dailies"]/div/div[2]/div[2]/div[2]/div[2]/div[2]/div[3]/div/div[3]/div/div[2]/div/div/div[2]/button[3]'
    },
    'sampling_period[min]': {
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

dme_root_files = f"CellEnMon/datasets/dme/{start_date_str_rep_ddmmyyyy}_{end_date_str_rep_ddmmyyyy}"  # DD/MM/YYYY
dme_root_files_processed = f"{dme_root_files}/processed"
dme_root_files_raw = f"{dme_root_files}/raw"
dme_root_files_paths = f"{dme_root_files}/paths"
create_directory_if_does_not_exist(dme_root_files_processed)
create_directory_if_does_not_exist(dme_root_files_raw)
create_directory_if_does_not_exist(dme_root_files_paths)


dme_scrape_config = {
    'url': 'http://tau.omnisol.co.il/',
    'link_objects': {
        'date': {
            'select': 'In range',

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            'value': start_date_str_rep_mmddyyyy,
            'value_range': end_date_str_rep_mmddyyyy
        },
        'measurement_type': [],  # ['TN_RFInputPower'],
        'data_precentage': {
            'select': 'Equals',
            'value': '100',

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'
        },
        'sampling_period_description': {},
        'sampling_period[min]': '15',

        'link_id': [
             'c409-5077', 'b219-5060', 'c219-5079','d409-5079',  # Arad: #d219-5079
            '803b-6879', 'a459-6879', 'a690-6880', 'b690-6881', 'a273-6881', 'b459-6880',  # Naot Smadar
            'c247-7049',  # Yavne
            'a473-5512', 'b119-5512',  # Paran
            'e032-5090', 'a247-5090', 'b247-5377',  # Ashdod
            'c394-7336', 'ts02-7332', 'ts06-7336', 'b394-7333', 'ts03-7331', 'a394-7332',  # Bar Shava
            'h086-7193', 'g086-5091'  # Ashkelon
        ],

        'link_carrier': {},
        'link_frequency[mhz]': {
            # Equal
            # Not equal
            # Less than
            # Less than or equals
            # Greater than
            # Greater than or equals
            # In range

            'select': 'Not equal',
            'value': '0'
        },
        'link_polarization': {},
        'link_length[km]': {},
        'link_expired_on': {},
        'link_antenna_height[m]': {},
        'tx_site_id': {},
        'tx_site_secondary_id': {},
        'tx_site_longitude': {
            'select': None,

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            # 'value': '34.088504434511606',
            # 'value_range': '35.630255532834326'
        },
        'tx_site_latitude': {
            'select': None,

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            # 'value': '29.416005191321844',
            # 'value_range': '31.430796575396656'
        },
        'tx_site_tower_hight[m]': {},
        'rx_site_id': {},
        'rx_site_secondary_id': {},
        'rx_site_longitude': {
            'select': None,

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            # 'value': '34.088504434511606',
            # 'value_range': '35.630255532834326'
        },
        'rx_site_latitude': {
            'select': None,

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            # 'value': '29.416005191321844',
            # 'value_range': '31.430796575396656'
        },
        'rx_site_tower_height[m]': {},

        'hop_id': {},
        'hop_length[km]': {},
        'hop_status': {}
    }
}


######################################
########## DATE VALIDATION ###########
######################################

def parse_date(d):
    return d['mm'] + '/' + d['dd'] + '/' + d['yyyy'][-2:]


value = parse_date(date['value']).strip()
value_range = parse_date(date['value_range']).strip()
date_select = dme_scrape_config['link_objects']['date']['select']

if not value or date_select == 'In range' and dt.strptime(value, "%m/%d/%y") > dt.strptime(value_range, "%m/%d/%y"):
    raise ValueError('missing value in date, or value_range is earlier than value')

coverage = (dt.strptime(value_range, "%m/%d/%y") - dt.strptime(value, "%m/%d/%y")).days + 1

###############################################
########## EVALUATE a,b,L POWER-LAW ###########
###############################################
basic_db_path = 'libs/power_law/frequency_dependent_coefficients_for_estimating_specific.csv'

########################################################################################################################
###########################################    LEARNING   ##############################################################
########################################################################################################################


################################
########## EXTRACTOR ###########
################################

# IMS
ims_metadata = ['latitude', 'longitude']

# DME
dme_metadata = {
    # 'frequency': 'Link Frequency [MHz]',
    # 'polarization': 'Link Polarization',
    # 'length': 'Link Length (KM)',
    'tx_longitude': 'Tx Site Longitude',
    'tx_latitude': 'Tx Site Latitude',
    'rx_longitude': 'Rx Site Longitude',
    'rx_latitude': 'Rx Site Latitude',
    # 'carrier': 'Link Carrier',
    # 'id': 'Link ID'
}
