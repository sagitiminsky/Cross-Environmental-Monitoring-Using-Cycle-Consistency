#####
# export PYTHONPATH=$PYTHON_PATH:/Users/sagit/Desktop/CellEnMon-Research
from datetime import datetime as dt
from datetime import timedelta as dt_delta

MAC = True
download_path = '/Users/sagitiminsky/Downloads' if MAC == True else '/home/sagit/Downloads'


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
        'year_str_rep': f"{res.year}",
        'str_rep_ddmmyyyy': f"{str(res.day).zfill(2)}{str(res.month).zfill(2)}{res.year}"
    }


date = {
    'value': {
        'dd': '01',
        'mm': '01',
        'yyyy': '2013'
    },
    'value_range': {
        'dd': '31',
        'mm': '12',
        'yyyy': '2015'
    }
}

date_str_rep = add_days_to_date(date['value'])['str_rep_with_replace'] + '_' + add_days_to_date(date['value_range'])[
    'str_rep_with_replace']
date_datetime_rep = add_days_to_date(date['value'])['datetime_rep']
start_date_str_rep = add_days_to_date(date['value'])['str_rep_ddmmyyyy']
end_date_str_rep = add_days_to_date(date['value_range'])['str_rep_ddmmyyyy']

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

ims_root_files = f"datasets/ims/raw/{start_date_str_rep}_{end_date_str_rep}"  # DD/MM/YYYY
ims_root_values = 'datasets/ims/processed'
ims_token = 'f058958a-d8bd-47cc-95d7-7ecf98610e47'
ims_mapping = [
    '241',
    '348',
    '202',
    '10',
    '106',
    '73',
    '353',
    '343',
    '62',
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
    '64'
]

ims_scrape_config = {
    '_from': f"{add_days_to_date(date['value'])['str_rep']}",  # MM/DD/YYYY
    '_to': f"{add_days_to_date(date['value_range'])['str_rep']}",  # MM/DD/YYYY , delta_days=2

    'left_bound': 1 - 1,
    'right_bound': len(ims_mapping)
}

################################
######### DME SCRAPPER #########
################################

xpaths = {
    'xpath_download': '//*[@id="btnExport"]',
    'xpath_metadata_download': '//*[@id="btnExportMetadata"]',
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

dme_root_files = f"datasets/dme/raw/{start_date_str_rep}_{end_date_str_rep}"  # DD/MM/YYYY
dme_paths_root = './libs/scrappers/dme_scrapper/paths'
dme_root_values = 'datasets/dme/processed'
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

            'value': start_date_str_rep,
            'value_range': end_date_str_rep
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
            'd409-5079', 'c409-5077', 'b219-5060', 'c219-5079',  # Arad - no data: d219-5079
            '803b-6879', 'a459-6879', 'a690-6880', 'b690-6881', 'a273-6881', 'b459-6880',  # Naot Smadar - no data:
            'c247-7049',  # Yavne - no data:
            'a473-5512', 'b119-5512',  # Paran - no data:
            'e032-5090', 'a247-5090', 'b247-5377',  # Ashdod - no data:
            'c394-7336', 'ts02-7332', 'ts06-7336', 'b394-7333',  # Bar Shava - no data: ts03-7331, a394-7332
            'g086-5091'  # Ashkelon - no data: h086-7193
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
ims_pre_load_data = True
ims_metadata = ['latitude', 'longitude']

# DME
dme_pre_load_data = True
dme_metadata = {
    'frequency': 'Link Frequency [MHz]',
    'polarization': 'Link Polarization',
    'length': 'Link Length (KM)',
    'tx_longitude': 'Tx Site Longitude',
    'tx_latitude': 'Tx Site Latitude',
    'rx_longitude': 'Rx Site Longitude',
    'rx_latitude': 'Rx Site Latitude',
    'carrier':'Link Carrier',
    'id':'Link ID'
}
