import math
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
import models
import wandb

ENABLE_WANDB = False
GROUPS = {
    "DEBUG": {0: "DEBUG"},
    "DYNAMIC_ONLY": {0: "lower metrics", 1: "without RR", 2: "with RR and inv_dist", 3: "with RR only"}
}
SELECTED_GROUP_NAME = "DYNAMIC_ONLY"
SELECT_JOB = 3

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

    for epoch in range(train_opt.epoch_count,
                       train_opt.n_epochs + train_opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        agg_train_mse_A, agg_train_mse_B = 0, 0
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            # TODO: on setinput we currently take only the data, we should consider using the metadata too
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
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, train_opt.n_epochs + train_opt.n_epochs_decay, time.time() - epoch_start_time))

        if epoch % 10 == 0:
            # Validation losses
            agg_validation_mse_A, agg_validation_mse_B = 0, 0
            for val_data in validation_dataset:
                model.set_input(val_data)
                model.optimize_parameters(is_train=False)  # calculate loss functions
                validation_losses = model.get_current_losses(is_train=False)
                agg_validation_mse_A += validation_losses["Validation/mse_A"]
                agg_validation_mse_B += validation_losses["Validation/mse_B"]

            print(f"Training Losses:{training_losses}\n\n")
            print(f"Validation Losses:{validation_losses}\n\n")

            # agg fix
            validation_losses['Validation/rmse_A'] = math.sqrt(agg_validation_mse_A / len(validation_dataset))
            validation_losses['Validation/rmse_B'] = math.sqrt(agg_validation_mse_B / len(validation_dataset))
            training_losses['Train/rmse_A'] = math.sqrt(agg_train_mse_A / len(train_dataset))
            training_losses['Train/rmse_B'] = math.sqrt(agg_train_mse_B / len(train_dataset))

            if ENABLE_WANDB:
                wandb.log({**validation_losses, **training_losses})

