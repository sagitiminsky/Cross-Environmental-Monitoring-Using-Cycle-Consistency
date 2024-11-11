"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from CellEnMon.data.base_dataset import BaseDataset, get_transform
from .exporter import Extractor
import random
import torch
import numpy as np
from math import radians, cos, sin, asin, sqrt
import torch.nn.functional as F
from datetime import datetime
import config
import os

# from data.image_folder import make_dataset
# from PIL import Image

# DIRECTIONS
LEFT = (1, 0, 0, 0)
RIGHT = (0, 1, 0, 0)
UP = (0, 0, 1, 0)
DOWN = (0, 0, 0, 1)

LAMBDA=float(os.environ["LAMBDA"])


class CellenmonDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super().__init__(opt)
        self.dataset = Extractor(is_train=opt.isTrain)
        self.dataset.stats()  # get a,b,c for a * np.exp(-b * x) + c

        self.dme_len = len(self.dataset.dme.db)
        self.ims_len = len(self.dataset.ims.db)

    def pad_with_respect_to_direction(self, A, B, dir, value_a, value_b):
        A = F.pad(input=A, pad=dir, mode='constant', value=value_a)
        B = F.pad(input=B, pad=dir, mode='constant', value=value_b)
        return A, B

    def calc_dist_and_center_point(self, x1_longitude, x1_latitude, x2_longitude, x2_latitude) -> dict:
        
        # Convert latitude and longitude from degrees to radians
        lat1 = radians(x1_latitude)
        lat2 = radians(x2_latitude)
        lon1 = radians(x1_longitude)
        lon2 = radians(x2_longitude)

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))

        # Radius of Earth in kilometers
        r = 6371
        distance = c * r

        # Calculate the midpoint
        midpoint_longitude = (x1_longitude + x2_longitude) / 2
        midpoint_latitude = (x1_latitude + x2_latitude) / 2

        return {
            "dist": distance,  # in km
            "center": {
                "longitude": midpoint_longitude,
                "latitude": midpoint_latitude
            }
        }

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary."""
        
        entry_list_dme=list(self.dataset.dme.db_normalized)
        entry_list_ims=list(self.dataset.ims.db_normalized)
        filter_cond=True
        while filter_cond:
            
            selected_link = entry_list_dme[random.randint(0, self.dme_len - 1)]
            data_dict_A = self.dataset.dme.db_normalized[selected_link]

            selected_gague = entry_list_ims[random.randint(0, self.ims_len - 1)]
            data_dict_B = self.dataset.ims.db_normalized[selected_gague]  # needs to be a tensor

            link_metadata = self.dataset.dme.db[selected_link]['metadata']
            gauge_metadata=self.dataset.ims.db[selected_gague]['metadata']


            dme_station_coo = self.calc_dist_and_center_point(x1_longitude=link_metadata[0],
                                                          x1_latitude=link_metadata[1],
                                                          x2_longitude=link_metadata[2],
                                                          x2_latitude=link_metadata[3])


            dist = self.calc_dist_and_center_point(x1_longitude=gauge_metadata[1],
                                               x1_latitude=gauge_metadata[0],
                                               x2_longitude=dme_station_coo["center"]["longitude"],
                                               x2_latitude=dme_station_coo["center"]["latitude"])["dist"]        


            if config.TRAIN_RADIUS >= dist:
                # print("link and gauge are in radius")
                slice_start_A = 0
                slice_start_B = 0
                slice_dist = self.opt.slice_dist
                time_stamp_A_start_time = 0
                time_stamp_B_start_time = 1
                dme_vec_len = len(data_dict_A['data'])
                ims_vec_len = len(data_dict_B['data'])
                


                slice_start_A = random.randint(0, dme_vec_len - 1)
                time_stamp_A_start_time = list(data_dict_A['data'].keys())[slice_start_A]
                
                if time_stamp_A_start_time[:10] in [x[:10] for x in data_dict_B['data']]: # 13:dutch | 10:israel
                    for l in data_dict_B['data'].keys():
                        if l.startswith(time_stamp_A_start_time[:10]) and \
                        slice_start_A + slice_dist <= dme_vec_len and\
                        slice_start_B + slice_dist <= ims_vec_len:
                            slice_start_B=list(data_dict_B['data'].keys()).index(l)
                            filter_cond=False
        
        slice_end_A = slice_start_A + slice_dist
        slice_end_B = slice_start_B + slice_dist
        
        # print(f"Selected dist is: {dist}")

        A = torch.Tensor(np.array(list(data_dict_A['data'].values())[slice_start_A:slice_end_A]))
        B = torch.Tensor(np.array(list(data_dict_B['data'].values())[slice_start_B:slice_end_B]))
        
        if self.opt.is_only_dynamic:
            A = A.T
            B = B.unsqueeze(0)

        assert(A.shape==((4,self.opt.slice_dist)))
        assert(B.shape==((1,self.opt.slice_dist)))


        if "DEBUG" in os.environ and int(os.environ["DEBUG"]):
            print(f"A: {A}")
            print(f"B: {B}")
        # else:
            
        #     A=A.reshape(self.opt.slice_dist,4)
        #     B=B.reshape(self.opt.slice_dist,1)
        #     for a, b in zip(data_dict_A['norm_metadata'], data_dict_B['norm_metadata']):
        #         A, B = self.pad_with_respect_to_direction(A, B, RIGHT, value_a=a, value_b=b)

        #     A=A.T # (8,256)
        #     B=B.T # (5,256)

        A_attenuation=self.min_max_inv_transform(x=A,mmin=-50.8,mmax=17)
        B_rain_rate = self.min_max_inv_transform(x=B,mmin=0,mmax=3.3)
        
        a=LAMBDA

        return {
            'A': A,
            'B': B,
            'Time': list(data_dict_A['data'].keys())[slice_start_A:slice_end_A],
            'link': selected_link,
            'link_norm_metadata': data_dict_A["norm_metadata"],
            'link_metadata':link_metadata,
            'link_full_name': f'{selected_link.split("-")[0]}-{link_metadata[0]}-{link_metadata[1]}-{selected_link.split("-")[1]}-{link_metadata[2]}-{link_metadata[3]}',
            'link_center_metadata': dme_station_coo["center"],
            'gague': selected_gague,
            'gague_norm_metadata': data_dict_B["norm_metadata"],
            'gague_metadata': gauge_metadata,
            'gague_full_name': f'{selected_gague}-{gauge_metadata[0]}-{gauge_metadata[1]}',
            'metadata_A': link_metadata,
            'metadata_B': gauge_metadata,
            'data_transformation': {'link': {'min': data_dict_A['data_min'], 'max': data_dict_A['data_max']},
                                    'gague': {'min': data_dict_B['data_min'], 'max': data_dict_B['data_max']}},
            'metadata_transformation': {'metadata_lat_max': self.dataset.metadata_lat_max,
                                        'metadata_lat_min': self.dataset.metadata_lat_min,
                                        'metadata_long_max': self.dataset.metadata_long_max,
                                        'metadata_long_min': self.dataset.metadata_long_min},
            'distance': dist,  # in KM
            
            'attenuation_prob': self.func_fit(x=A_attenuation, a=a),
            'rain_rate_prob': self.func_fit(x=B_rain_rate, a=a),
            
#             'a_rain': self.dataset.a,
#             'b_rain': self.dataset.b,
#             'c_rain': self.dataset.c
        }

    def min_max_inv_transform(self,x, mmin, mmax):
        return x #x * (mmax - mmin) + mmin
    
    def func_fit(self, x, a):
        x=torch.from_numpy(np.array(x))
        b=torch.from_numpy(np.array(a))
        return torch.exp(a*x**2)

    def __len__(self):
        """We do 1000 random selects between CML and Gauge"""
        return int(os.environ["NUMBER_OF_CML_GAUGE_RANDOM_SELECTIONS_IN_EACH_EPOCH"])
