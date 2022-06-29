import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
import models
import wandb
ENABLE_WANDB=True
JOB_TYPES={0:"DEBUG"}

if __name__ == '__main__':
    train_opt = TrainOptions().parse()  # get training options
    validation_opt = TestOptions().parse()
    if ENABLE_WANDB:
        wandb.init(project=train_opt.name,  entity='sagitiminsky', job_type=JOB_TYPES[0], group=f"exp_{JOB_TYPES[0]}")
    train_dataset = data.create_dataset(train_opt)  # create a train dataset given opt.dataset_mode and other options
    validation_dataset = data.create_dataset(validation_opt)  # create a train dataset given opt.dataset_mode and other options
    model = models.create_model(train_opt)  # create a model given opt.model and other options
    model.setup(train_opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations

    for epoch in range(train_opt.epoch_count,
                       train_opt.n_epochs + train_opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            #TODO: on setinput we currently take only the data, we should consider using the metadata too
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % train_opt.print_freq == 0:  # print training and validation losses and save logging information to the disk
                # Training losses
                training_losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / train_opt.batch_size

                # Validation losses
                model.set_input(validation_dataset[i % len(validation_dataset)])
                validation_losses = model.get_current_losses()


                print(f"Training Losses:{training_losses}\n\n")
                print(f"Validation Losses:{validation_losses}\n\n")
                columns = ['train', 'validation']
                if ENABLE_WANDB:
                    for loss_name in training_losses:
                        wandb.log({
                            f"Train/{loss_name}": training_losses[loss_name],
                            f"Validation/{loss_name}": training_losses[loss_name],
                            "x": epoch
                        })

            if total_iters % train_opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if train_opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        # if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, train_opt.n_epochs + train_opt.n_epochs_decay, time.time() - epoch_start_time))
