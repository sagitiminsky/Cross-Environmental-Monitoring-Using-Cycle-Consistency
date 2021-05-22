###
# export PYTHONPATH=$PYTHON_PATH:/Users/sagit/Desktop/CellEnMon-Research
from datetime import datetime as dt
from datetime import timedelta as dt_delta

download_path = '/Users/sagitiminsky/Downloads/'


def parse_date(d):
    return d['mm'] + '/' + d['dd'] + '/' + d['yyyy'][-2:]


def add_days_to_date(date, delta_days=0):
    res = dt.strptime(parse_date(date), "%m/%d/%y") + dt_delta(days=delta_days)
    return {
        'datetime_rep': res,
        'str_rep': f"{str(res.month).zfill(2)}/{str(res.day).zfill(2)}/{res.year}",
        'str_rep_with_replace': f"{str(res.month).zfill(2)}{str(res.day).zfill(2)}{res.year}",
        'day_str_rep': f"{str(res.day).zfill(2)}",
        'month_str_rep': f"{str(res.month).zfill(2)}",
        'year_str_rep': f"{res.year}"
    }


date = {
    'value': {
        'dd': '01',
        'mm': '01',
        'yyyy': '2019'
    },
    'value_range': {
        'dd': '02',
        'mm': '01',
        'yyyy': '2019'
    }
}

date_str_rep = add_days_to_date(date['value'])['str_rep_with_replace'] + '_' + add_days_to_date(date['value_range'])[
    'str_rep_with_replace']
date_datetime_rep = add_days_to_date(date['value'])['datetime_rep']

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

#######################################
############## IMS SCRAPPER ###########
#######################################

ims_root_files = f"CellEnMon/libs/relics/datasets/ims/raw/{add_days_to_date(date['value'])['str_rep_with_replace']}_{add_days_to_date(date['value_range'])['str_rep_with_replace']}"  # MM/DD/YYYY'
ims_root_values = 'CellEnMon/libs/relics/datasets/ims/processed'
ims_token = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
ims_mapping = [
    '241', '348', '202', '10', '106', '73', '353', '343', '62',
    '269',
    '123',
    '227',
    '205',
    '233',
    '6',
    '99',
    '78',
    '46',
    '115',
    '2',
    '41',
    '43',
    '42',
    '186',
    '13',
    '8',
    '11',
    '355',
    '44',
    '264',
    '67',
    '16',
    '45',
    '263',
    '380',
    '224',
    '46',
    '206',
    '366',
    '107',
    '20',
    '90',
    '275',
    '270',
    '21',
    '178',
    '54',
    '30',
    '24',
    '124',
    '259',
    '74',
    '228',
    '121',
    '188',
    '23',
    '218',
    '22',
    '274',
    '75',
    '25',
    '77',
    '82',
    '208',
    '236',
    '210',
    '79',
    '211',
    '350',
    '28',
    '58',
    '59',
    '29',
    '349',
    '112',
    '65',
    '98',
    '338',
    '271',
    '33',
    '379',
    '207',
    '232',
    '36',
    '64']
ims_scrape_config = {
    '_from': f"{add_days_to_date(date['value'])['str_rep']}",  # MM/DD/YYYY
    '_to': f"{add_days_to_date(date['value_range'])['str_rep']}",  # MM/DD/YYYY , delta_days=2

    'left_bound': 1 - 1,
    'right_bound': len(ims_mapping)
}

################################
######### DME SCRAPPER #########
################################

dme_root_files = f"CellEnMon/libs/relics/datasets/dme/raw/{add_days_to_date(date['value'])['str_rep_with_replace']}_{add_days_to_date(date['value_range'])['str_rep_with_replace']}/"  # MM/DD/YYYY
dme_root_values = 'CellEnMon/libs/relics/datasets/dme/processed'
dme_scrape_config = {
    'username': 'SagiT',
    'password': 'W@st2020',
    'url': 'http://tau.omnisol.co.il/',
    'link_objects': {
        'date': {
            'select': 'In range',

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            'value': {
                'dd': f'{date["value"]["dd"]}',
                'mm': f'{date["value"]["mm"]}',
                'yyyy': f'{date["value"]["yyyy"]}'
            },
            'value_range': {
                'dd': f'{date["value_range"]["dd"]}',
                'mm': f'{date["value_range"]["mm"]}',
                'yyyy': f'{date["value_range"]["yyyy"]}'
            }

        },
        'measurement_type': 'TN_RFInputPower',
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
        'sampling_period[sec]': '15',

        'link_id': [],  # ['B394-D431'],

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
            'select': 'In range',

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            'value': '34.088504434511606',
            'value_range': '35.630255532834326'
        },
        'tx_site_latitude': {
            'select': 'In range',

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            'value': '29.416005191321844',
            'value_range': '31.430796575396656'
        },
        'tx_site_tower_hight[m]': {},
        'rx_site_id': {},
        'rx_site_secondary_id': {},
        'rx_site_longitude': {
            'select': 'In range',

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            'value': '34.088504434511606',
            'value_range': '35.630255532834326'
        },
        'rx_site_latitude': {
            'select': 'In range',

            # 'Equals'
            # 'Greater than'
            # 'Less than'
            # 'Not equal'
            # 'In range'

            'value': '29.416005191321844',
            'value_range': '31.430796575396656'
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


value = parse_date(dme_scrape_config['link_objects']['date']['value']).strip()
value_range = parse_date(dme_scrape_config['link_objects']['date']['value_range']).strip()
date_select = dme_scrape_config['link_objects']['date']['select']

if not value or date_select == 'In range' and dt.strptime(value, "%m/%d/%y") > dt.strptime(value_range, "%m/%d/%y"):
    raise ValueError('missing value in date, or value_range is earlier than value')

coverage = (dt.strptime(value_range, "%m/%d/%y") - dt.strptime(value, "%m/%d/%y")).days + 1

###############################################
########## EVALUATE a,b,L POWER-LAW ###########
###############################################
basic_db_path = 'CellEnMon/libs/power_law/frequency_dependent_coefficients_for_estimating_specific.csv'

########################################################################################################################
###########################################    LEARNING   ##############################################################
########################################################################################################################


################################
########## EXTRACTOR ###########
################################

# IMS
ims_pre_load_data = False
ims_metadata = ['latitude', 'longitude']

# DME
dme_pre_load_data = False
dme_metadata=['frequency', 'polarization', 'length', 'txsite_longitude', 'txsite_latitude', 'rxsite_longitude',
                     'rxsite_latitude']

