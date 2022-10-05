import math
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
import models
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mpl_dates

plt.switch_backend('agg')  # RuntimeError: main thread is not in main loop

ENABLE_WANDB = True
GROUPS = {
    "DEBUG": {0: "DEBUG"},
    "DYNAMIC_ONLY": {0: "lower metrics", 1: "without RR", 2: "with RR and inv_dist", 3: "with RR only"},
    "Dymanic and Static": {0: "first try", 1: "with RR only", 2: "plot with un-norm. values"},
    "Dymanic and Static 4x1 <-> 1x4": {0: "first try"},
    "Dynamic only": {0: "first try"},
    "Dynamic and Static": {0: "first try"}
}

SELECTED_GROUP_NAME = "Dynamic only"
SELECT_JOB = 0




DME_KEYS = {1: 'TMmax[dBm]', 2: 'TMmin[dBm]', 3: 'RMmax[dBm]', 4: 'RMmin[dBm]'}
IMS_KEYS = {1: 'RR[mm/h]'}


def toggle(t):
    if t == 'AtoB':
        return 'BtoA'
    else:
        return 'AtoB'


def min_max_inv_transform(x, mmin, mmax):
    return x * (mmax - mmin) + mmin


if __name__ == '__main__':
    train_opt = TrainOptions().parse()  # get training options
    validation_opt = TestOptions().parse()
    if ENABLE_WANDB:
        wandb.init(project=train_opt.name, entity='sagitiminsky',
                   group=f"exp_{SELECTED_GROUP_NAME}", job_type=GROUPS[SELECTED_GROUP_NAME][SELECT_JOB])
    train_dataset = data.create_dataset(train_opt)  # create a train dataset given opt.dataset_mode and other options
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

                visuals = model.get_current_visuals()
                fig, axs = plt.subplots(2, 3, figsize=(15, 15))
                title = f'{model.link}<->{model.gague}' if train_opt.is_only_dynamic else f'{model.link}<->{model.gague}' \
                                                                                          f'[T.long, T.lat, R.long, R.lat]'
                plt.title(title)

                for ax, key in zip(axs.flatten(), visuals):

                    if train_opt.is_only_dynamic:
                        N = 4 if 'A' in key else 1
                    else:
                        raise NotImplemented

                    # Plot Data
                    data = visuals[key][0].reshape(256, N).cpu().detach().numpy()
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

                        if not train_opt.is_only_dynamic:
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

                            metadata = ["{:.3f}".format(min_max_inv_transform(x, mmin=mmin, mmax=mmax)) for
                                        (mmin, mmax), x in
                                        zip(metadata_inv_zip, visuals[key][0][0, 4:8].cpu().detach().numpy())]

                        ax.plot([mpl_dates.date2num(datetime.strptime(t[0], '%Y-%m-%d %H:%M:%S')) for t in model.t],
                                min_max_inv_transform(data_vector, mmin=mmin, mmax=mmax),
                                marker='o',
                                linestyle='dashed',
                                linewidth=0.0,
                                markersize=4,
                                label=label
                                )

                        ax.set_title(key if train_opt.is_only_dynamic else f'{key} \n'
                                                                           f' {metadata}', y=0.75)

                    # Formatting Date
                    date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M:%S')
                    ax.xaxis.set_major_formatter(date_format)

                wandb.log({**validation_losses, **training_losses})
                # wandb.log({"Images": [wandb.Image(visuals[key], caption=key) for key in visuals]})
                wandb.log({title: plt})
