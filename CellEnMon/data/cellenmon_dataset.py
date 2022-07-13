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

# from data.image_folder import make_dataset
# from PIL import Image

# DIRECTIONS
LEFT = (1, 0, 0, 0)
RIGHT = (0, 1, 0, 0)
UP = (0, 0, 1, 0)
DOWN = (0, 0, 0, 1)


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
        lon1 = radians(x1_longitude)
        lon2 = radians(x1_latitude)
        lat1 = radians(x2_longitude)
        lat2 = radians(x2_latitude)

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers (6371). Use 3956 for miles. We use 1 because it is normalized
        r = 1
        return {
            "dist": c * r,
            "center": {
                "longitude": (x1_longitude + x2_longitude) / 2,
                "latitude": (x1_latitude + x2_latitude) / 2
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
        Step 4: return a data point as a dictionary.
        """
        entry_list_dme = list(self.dataset.dme.db_normalized)
        entry_list_ims = list(self.dataset.ims.db_normalized)
        data_dict_A = self.dataset.dme.db_normalized[entry_list_dme[index % self.dme_len]]
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.ims_len
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.ims_len - 1)

        data_dict_B = self.dataset.ims.db_normalized[entry_list_ims[index_B % self.ims_len]]  # needs to be a tensor

        """
        dme: a day contains 96 samples
        ims: a day contains 144 samples
        
        So that after taking each 2nd measurment of dme
        And after taking each 3rd measurment of ims we get
        
        """
        ##########################
        #### Station Distance ####
        ##########################

        dme_station_coo = self.calc_dist_and_center_point(x1_longitude=data_dict_A["metadata"][0],
                                                          x1_latitude=data_dict_A["metadata"][1],
                                                          x2_longitude=data_dict_A["metadata"][2],
                                                          x2_latitude=data_dict_A["metadata"][3])

        ims_station_coo = self.calc_dist_and_center_point(x1_longitude=data_dict_B["metadata"][0],
                                                          x1_latitude=data_dict_B["metadata"][1],
                                                          x2_longitude=data_dict_B["metadata"][0],
                                                          x2_latitude=data_dict_B["metadata"][1])

        dist = self.calc_dist_and_center_point(x1_longitude=ims_station_coo["center"]["longitude"],
                                               x1_latitude=ims_station_coo["center"]["latitude"],
                                               x2_longitude=dme_station_coo["center"]["longitude"],
                                               x2_latitude=dme_station_coo["center"]["latitude"])["dist"]

        ################################
        #### Rain Rate Significance ####
        ################################

        slice_start_A = 0
        slice_start_B = 0
        slice_dist = self.opt.slice_dist
        time_stamp_A_start_time = 0
        time_stamp_B_start_time = 1
        dme_vec_len = len(data_dict_A['data'])
        ims_vec_len = len(data_dict_B['data'])
        filter_cond = True

        while filter_cond:
            # go fetch
            slice_start_A = random.randint(0, dme_vec_len - 1)
            time_stamp_A_start_time = list(data_dict_A['data'].keys())[slice_start_A]

            if time_stamp_A_start_time in data_dict_B['data']:
                slice_start_B = list(data_dict_B['data'].keys()).index(time_stamp_A_start_time)

                filter_cond = slice_start_A + slice_dist > dme_vec_len \
                              or slice_start_B + slice_dist > ims_vec_len
        slice_end_A = slice_start_A + slice_dist
        slice_end_B = slice_start_B + slice_dist

        A = torch.Tensor(np.array(list(data_dict_A['data'].values())[slice_start_A:slice_end_A]))
        B = torch.Tensor(np.tile(np.array(list(data_dict_B['data'].values())[slice_start_B:slice_end_B]), (4, 1)).T)

        if self.opt.is_only_dynamic:
            A = A.repeat(1, 64).reshape(1, 256, 256)
            B = B.repeat(1, 64).reshape(1, 256, 256)
        else:
            A_LEFT, B_LEFT = self.pad_with_respect_to_direction(A, B, LEFT, value_a=data_dict_A['metadata'][0], value_b=data_dict_B['metadata'][0])
            A_UP, B_UP = self.pad_with_respect_to_direction(A_LEFT, B_LEFT, UP, value_a=data_dict_A['metadata'][2],
                                                            value_b=data_dict_B['metadata'][1])
            A_RIGHT, B_RIGHT = self.pad_with_respect_to_direction(A_UP, B_UP, RIGHT,value_a=data_dict_A['metadata'][1], value_b=data_dict_B['metadata'][0])

            A, B = self.pad_with_respect_to_direction(A_RIGHT, B_RIGHT, DOWN,value_a=data_dict_A['metadata'][3], value_b=data_dict_B['metadata'][1])

            A = A.repeat(43,43).reshape(-1)[:256*256].reshape(1,256,256)
            B = B.repeat(43,43).reshape(-1)[:256*256].reshape(1, 256, 256)

        return {
            'A': A,
            'B': B,
            'Time_A': list(data_dict_A['data'].keys())[slice_start_A:slice_end_A],
            'Time_B': list(data_dict_B['data'].keys())[slice_start_B:slice_end_B],
            'metadata_A': data_dict_A['metadata'],
            'metadata_B': data_dict_B['metadata'],
            'distance': dist,  # in KM
            'rain_rate': self.func_fit(x=np.average(B), a=self.dataset.a, b=self.dataset.b, c=self.dataset.c)
        }

    def func_fit(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def __len__(self):
        """Return the total number of images."""
        return max(self.dme_len, self.ims_len)
