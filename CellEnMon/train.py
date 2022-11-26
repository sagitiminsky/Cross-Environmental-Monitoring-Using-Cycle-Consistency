import math
import os.path
import time

import pandas as pd

from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
import models
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mpl_dates
import config
import numpy as np
from libs.visualize.visualize import Visualizer
plt.switch_backend('agg')  # RuntimeError: main thread is not in main loop

ENABLE_WANDB = True
GROUPS = {
    "DEBUG": {0: "DEBUG"},
    "DYNAMIC_ONLY": {0: "lower metrics", 1: "without RR", 2: "with RR and inv_dist", 3: "with RR only"},
    "Dymanic and Static": {0: "first try", 1: "with RR only", 2: "plot with un-norm. values"},
    "Dymanic and Static 4x1 <-> 1x4": {0: "first try"},
    "Dynamic only": {0: "first try"},
    "Dynamic and Static": {0: "first try", 1:"play around with configurations"},
    "Dynamic and Static Dutch": {0: "first try", 1:"play around with configurations", 2:"real fake gauge metric", 3:"make sure that fake gague is not too far from real link, and not too far from real gauge for validation"},
    "Dynamic and Static Israel": {0: "first try", 1:"play around with configurations"}
}

SELECTED_GROUP_NAME = "Dynamic and Static Dutch"
SELECT_JOB = 3




DME_KEYS = {1: 'TMmax[dBm]', 2: 'TMmin[dBm]', 3: 'RMmax[dBm]', 4: 'RMmin[dBm]'}
IMS_KEYS = {1: 'RR[mm/h]'}


def toggle(t):
    if t == 'AtoB':
        return 'BtoA'
    else:
        return 'AtoB'


def min_max_inv_transform(x, mmin, mmax):
    return (x+1) * (mmax - mmin) * 0.5 + mmin


if __name__ == '__main__':
    real_fake_gauge_metric={}
    datetime_format='%Y-%m-%d %H:%M' if config.export_type=="israel" else '%d-%m-%Y %H:%M' # no seconds required
    train_opt = TrainOptions().parse()  # get training options
    validation_opt = TestOptions().parse()
    experiment_name = "only_dynamic" if train_opt.is_only_dynamic else "dynamic_and_static"
    v = Visualizer(experiment_name=experiment_name)
    print("Visualizer Initialized!")
    if ENABLE_WANDB:
        wandb.init(project=train_opt.name, entity='sagitiminsky',
                   group=f"exp_{SELECTED_GROUP_NAME}", job_type=GROUPS[SELECTED_GROUP_NAME][SELECT_JOB])
    print(f'💪Train💪')
    train_dataset = data.create_dataset(train_opt)  # create a train dataset given opt.dataset_mode and other options
    print(f'\n\n👀Validation👀')
    validation_dataset = data.create_dataset(
        validation_opt)  # create a train dataset given opt.dataset_mode and other options


    model = models.create_model(train_opt)  # create a model given opt.model and other options
    model.setup(train_opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    direction = train_opt.direction
    for epoch in range(train_opt.epoch_count,
                       train_opt.n_epochs + train_opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        agg_train_mse_A, agg_train_mse_B = 0, 0
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

        print(f"Direction:{direction}")
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            
            
            
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(is_train=True)  # calculate loss functions, get gradients, update network weights
            
            # Training losses
            training_losses = model.get_current_losses(is_train=True)
            agg_train_mse_A += training_losses["Train/mse_A"]
            agg_train_mse_B += training_losses["Train/mse_B"]

            t_comp = (time.time() - iter_start_time) / train_opt.batch_size

            if total_iters % train_opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
                # model.save_networks(save_suffix)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, train_opt.n_epochs + train_opt.n_epochs_decay, time.time() - epoch_start_time))

        if epoch % 5 == 0:
            # Validation losses
            agg_validation_mse_A, agg_validation_mse_B = 0, 0
            for val_data in validation_dataset:
                model.set_input(val_data)
                model.optimize_parameters(is_train=False)  # calculate loss functions
                validation_losses = model.get_current_losses(is_train=False)
                agg_validation_mse_A += validation_losses["Validation/mse_A"]
                agg_validation_mse_B += validation_losses["Validation/mse_B"]
                
                

            # agg fix
            validation_losses['Validation/rmse_A'] = math.sqrt(agg_validation_mse_A / len(validation_dataset))
            validation_losses['Validation/rmse_B'] = math.sqrt(agg_validation_mse_B / len(validation_dataset))
            training_losses['Train/rmse_A'] = math.sqrt(agg_train_mse_A / len(train_dataset))
            training_losses['Train/rmse_B'] = math.sqrt(agg_train_mse_B / len(train_dataset))

            if ENABLE_WANDB:
                # Visualize
                plt.clf()
                metadata=[0]*4
                visuals = model.get_current_visuals()
                fig, axs = plt.subplots(2, 3, figsize=(15, 15))
                title = f'{model.link}<->{model.gague}' if train_opt.is_only_dynamic else f'{model.link}<->{model.gague}'

                plt.title(title)

                for ax, key in zip(axs.flatten(), visuals):

                    if train_opt.is_only_dynamic:
                        N = 4 if 'A' in key else 1
                    else:
                        N = 8 if 'A' in key else 5

                    # Plot Data
                    data = visuals[key][0].reshape(train_opt.slice_dist, N).cpu().detach().numpy() if train_opt.is_only_dynamic else visuals[key][0].T.cpu().detach().numpy()
                    for i in range(1, 5):
                        if 'A' in key:
                            mmin = model.data_transformation['link']['min'][0].numpy()
                            mmax = model.data_transformation['link']['max'][0].numpy()
                            label = DME_KEYS[i]
                            data_vector = data[:, i - 1]
                        else:
                            mmin = model.data_transformation['gague']['min'][0].numpy()
                            mmax = model.data_transformation['gague']['max'][0].numpy()
                            label = IMS_KEYS[1]
                            data_vector = data.T[0]


                        metadata_lat_max = float(model.metadata_transformation['metadata_lat_max'])
                        metadata_lat_min = float(model.metadata_transformation['metadata_lat_min'])
                        metadata_long_max = float(model.metadata_transformation['metadata_long_max'])
                        metadata_long_min = float(model.metadata_transformation['metadata_long_min'])

                        metadata_inv_zip = [
                            (metadata_long_max, metadata_long_min),
                            (metadata_lat_max, metadata_lat_min),
                            (metadata_long_max, metadata_long_min),
                            (metadata_lat_max, metadata_lat_min)
                        ]
                        metadata = ["{:.4f}".format(min_max_inv_transform(x, mmin=mmin, mmax=mmax)) for
                                    (mmin, mmax), x in
                                    zip(metadata_inv_zip, visuals[key][0][:,0][-4:].cpu().detach().numpy())]

                        ax.plot([mpl_dates.date2num(datetime.strptime(t[0], datetime_format)) for t in model.t],
                                min_max_inv_transform(data_vector, mmin=mmin, mmax=mmax),
                                marker='o',
                                linestyle='dashed',
                                linewidth=0.0,
                                markersize=4,
                                label=label
                                )

                        ax.set_title(key if train_opt.is_only_dynamic else f'{key} \n'
                                                                           f' {metadata}', y=0.75, fontdict={'fontsize':6})

                    # Formatting Date
                    date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M:%S')
                    ax.xaxis.set_major_formatter(date_format)

                    # save in predict directory for generated data
                    start_date = config.start_date_str_rep_ddmmyyyy
                    end_date = config.end_date_str_rep_ddmmyyyy
                    produced_gauge_folder = f'CellEnMon/datasets/dme/{start_date}_{end_date}/predict/{experiment_name}' if 'A' in key else f'CellEnMon/datasets/ims/{start_date}_{end_date}/predict/{experiment_name}'
                    real_gauge_folder=f'CellEnMon/datasets/ims/{start_date}_{end_date}/processed'

                    if not os.path.exists(produced_gauge_folder):
                        os.makedirs(produced_gauge_folder)
                        
                    if not os.path.exists(real_gauge_folder):
                        os.makedirs(real_gauge_folder)

                    if 'fake_B' == key:
                        file_path = f'{produced_gauge_folder}/PRODUCED_{model.link[0]}-{model.gague[0]}.csv'
                        with open(file_path, "w") as file:
                            a=np.array([t[0] for t in model.t]).reshape(train_opt.slice_dist,1)
                            b=min_max_inv_transform(data_vector, mmin=mmin, mmax=mmax).reshape(train_opt.slice_dist,1)
                            headers = ','.join(['Time']+list(DME_KEYS.values())) if 'A' in key else ','.join(['Time']+list(IMS_KEYS.values()))
                            c=np.hstack((a,b))
                            fmt = ",".join(["%s"]*(c.shape[1]))
                            np.savetxt(file, c, fmt=fmt, header=headers, comments='')
                            
                        # calculate metric for test gauges
                        real_fake_gauge_metric=0
                        counter=0
                        tested_with_array=[]
                        for real_gauge in v.real_gagues:

                            real_gauge_longitude=v.real_gagues[real_gauge]['Longitude']
                            real_gauge_latitude=v.real_gagues[real_gauge]['Latitude']

                            if v.is_within_radius(stations={
                                "fake_longitude":f'{float(metadata[0]):.3f}', 
                                "fake_latitude":f'{float(metadata[1]):.3f}',
                                "real_longitude": model.link_center_metadata['longitude'],
                                "real_latitude": model.link_center_metadata['latitude']},
                                radius=config.RADIUS) \
                            and v.is_within_radius(stations={
                                "fake_longitude":f'{float(metadata[0]):.3f}', 
                                "fake_latitude":f'{float(metadata[1]):.3f}',
                                "real_longitude":real_gauge_longitude,
                                "real_latitude":real_gauge_latitude},
                                radius=config.RADIUS): 

                                counter+=1
                                tested_with_array.append(real_gauge)

                                path_to_real_gauge=f"{real_gauge_folder}/{real_gauge}_{real_gauge_latitude}_{real_gauge_longitude}.csv"   
                                to_add=v.calculate_matric_for_real_and_fake_gauge(path_to_real_gauge=path_to_real_gauge,path_to_fake_gauge=file_path)

                                if to_add:
                                    real_fake_gauge_metric+=to_add
                                else:
                                    counter-=1

                        if tested_with_array:
                            print(f"👀   {model.link[0]}-{model.gague[0]} is validated with {tested_with_array}   👀")
                              
                                    
                            

                v.draw_cml_map(virtual_gauge_name=f'PRODUCED_{model.link[0]}-{model.gague[0]}.csv',virtual_gauge_coo={
                    "longitude": f'{model.link_center_metadata["longitude"][0]:.3f}' if train_opt.is_only_dynamic else f'{float(metadata[0]):.3f}',
                    "latitude": f'{model.link_center_metadata["latitude"][0]:.3f}' if train_opt.is_only_dynamic else f'{float(metadata[1]):.3f}'
                })
                
                
                path_to_html = f"{v.out_path}/{v.map_name}"
                wandb.log({**validation_losses, **training_losses, **{f"RMSE:{model.link[0]}-{model.gague[0]}": real_fake_gauge_metric**0.5 if counter!=0 else -10}})
                # wandb.log({"Images": [wandb.Image(visuals[key], caption=key) for key in visuals]})
                wandb.log({title: plt})
                wandb.log({"html": wandb.Html(open(path_to_html), inject=False)})