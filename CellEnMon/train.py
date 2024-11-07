import math
import os.path
import time
import torch
import pandas as pd
import glob
from options.train_options import TrainOptions
from options.test_options import TestOptions
import data
import models
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mpl_dates
import config
import torch.nn.functional as F
import numpy as np
from libs.visualize.visualize import Visualizer
from preprocess import Preprocess
from collections import OrderedDict
plt.switch_backend('agg')  # RuntimeError: main thread is not in main loop
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from matplotlib import colors
from collections import defaultdict

ENABLE_WANDB = bool(os.environ["ENABLE_WANDB"])
GROUPS = {
    "DEBUG": {0: "DEBUG"},
    "DYNAMIC_ONLY": {0: "lower metrics", 1: "without RR", 2: "with RR and inv_dist", 3: "with RR only"},
    "Dymanic and Static": {0: "first try", 1: "with RR only", 2: "plot with un-norm. values"},
    "Dymanic and Static 4x1 <-> 1x4": {0: "first try"},
    "Dynamic only": {0: "first try"},
    "Dynamic and Static": {0: "first try", 1:"play around with configurations"},
    "Dynamic and Static Dutch": {0: "first try", 1:"play around with configurations", 2:"real fake gauge metric", 3:"fake gague is not too positioning"},
    "Dynamic and Static Israel": {0: "first try", 1:"play around with configurations"},
    "Dynamic Dutch": {0:"first try"},
    "Real Validation": {0:"last try", 1:"last last try"},
    "Last Try":{0:"last last try"},
    "Frontiers":{0:"first try", 1:"start training on Israel data"},
    "Lahav":{0:"GAN+Cycle", 1:"Cycle", 2:"Cycle+Detector"},
    "Neot Smadar":{0:"GAN+Cycle",1:"Cycle", 2:"Cycle+Detector"},
    "1L1G": {0:"Cycle+Detector"},
    #...
    "9L6G": {0:"Cycle+Detector"},

}




# gauges in train dataset:dict_keys(['ZOMET HANEGEV', 'BEER SHEVA', 'EZUZ', 'NEOT SMADAR', 'SEDE BOQER', 'PARAN'])
# links in train dataset:dict_keys(['a477-b379', 'a459-803b', '462d-c088', 'a479-b477', 'b465-d481', 'b451-a350', 'b459-a690', 'b480-a458', 'f350-e483'])

# gauges in validation dataset:dict_keys(['NEVATIM', 'LAHAV'])
# links in validation dataset:dict_keys(['c078-d088', 'a473-b119', 'b394-ts04'])


# Validation Matching
all_link_to_gauge_matching ={
    "a479-b477": ["NEOT SMADAR","PARAN"],
    "a477-b379": ["NEOT SMADAR","PARAN"],
    "b459-a690": ["NEOT SMADAR","PARAN"],
    "a459-803b": ["NEOT SMADAR","PARAN"],
    "b480-a458": ["NEOT SMADAR","PARAN"],
    "a473-b119": ["NEOT SMADAR","PARAN"],
    
    "b465-d481": ["SEDE BOQER","EZUZ"],
    "c078-d088": ["SEDE BOQER","EZUZ"],
    "462d-c088": ["SEDE BOQER","EZUZ"],
    
    "b394-ts04": ["LAHAV", "NEVATIM", "BEER SHEVA", "SHANI" "ZOMET HANEGEV", "ZOVA"],
    "f483-ts05": ["LAHAV", "NEVATIM", "BEER SHEVA", "SHANI" "ZOMET HANEGEV", "ZOVA"],
    "f350-e483": ["LAHAV", "NEVATIM", "BEER SHEVA", "SHANI" "ZOMET HANEGEV", "ZOVA"],
    "b451-a350": ["LAHAV", "NEVATIM", "BEER SHEVA", "SHANI" "ZOMET HANEGEV", "ZOVA"],
    "b412-c349": ["LAHAV", "NEVATIM", "BEER SHEVA", "SHANI" "ZOMET HANEGEV", "ZOVA"],
    "a063-b349": ["LAHAV", "NEVATIM", "BEER SHEVA", "SHANI" "ZOMET HANEGEV", "ZOVA"],
    "d063-c409": ["LAHAV", "NEVATIM", "BEER SHEVA", "SHANI" "ZOMET HANEGEV", "ZOVA"],
    
    "j033-261c": ["NEGBA", "ASHQELON PORT", "NIZZAN"]
}

validation_link_to_gauge_matching ={
#     "c078-d088": [], 
#     "a473-b119": [], 
    "b394-ts04": ["LAHAV"], #
    "b459-a690": [], #"NEOT SMADAR"

}


### TO RUN:
# LAMBDA=2 SELECTED_GROUP_NAME="Lahav" SELECT_JOB=2 ITERS_BETWEEN_VALIDATIONS=1000 ENABLE_WANDB=True DEBUG=0 threshold=0.2 probability_threshold=0.5 python3 CellEnMon/train.py

# Environment Variables
threshold = float(os.environ["threshold"])
probability_threshold = float(os.environ["probability_threshold"]) #0.3 # a*e^(-bx)+c, ie. we consider a wet event over x=0.2 mm/h
ITERS_BETWEEN_VALIDATIONS=int(os.environ["ITERS_BETWEEN_VALIDATIONS"])
SELECTED_GROUP_NAME = os.environ["SELECTED_GROUP_NAME"]
SELECT_JOB = int(os.environ["SELECT_JOB"])
LAMBDA=int(os.environ["LAMBDA"])

# Detection:
#[[  52 2099]
#  [   3  150]]
# Regression:
#[[1793  358]
#  [ 126   27]]


#Formatting Date
date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M:%S')

DME_KEYS = {0: 'PowerTLTMmax[dBm]', 1: 'PowerTLTMmin[dBm]', 2: 'PowerRLTMmax[dBm]', 3: 'PowerRLTMmin[dBm]'}
IMS_KEYS = {0: 'RainAmout[mm/h]'}


def toggle(t):
    if t == 'AtoB':
        return 'BtoA'
    else:
        return 'AtoB'


def min_max_inv_transform(x, mmin, mmax):
    return x * (mmax - mmin) + mmin

# DIRECTIONS
LEFT = (1, 0, 0, 0)
RIGHT = (0, 1, 0, 0)
UP = (0, 0, 1, 0)
DOWN = (0, 0, 0, 1)

def func_fit(x, a):
    x=torch.from_numpy(np.array(x))
    b=torch.from_numpy(np.array(a))
    return 1/(a * torch.exp(-x*a))


def pad_with_respect_to_direction( A, B, dir, value_a, value_b):
    A = F.pad(input=A, pad=dir, mode='constant', value=value_a)
    B = F.pad(input=B, pad=dir, mode='constant', value=value_b)
    return A, B


if __name__ == '__main__':
    real_fake_gauge_metric={}
    dates_range = f"{config.start_date_str_rep_ddmmyyyy}_{config.end_date_str_rep_ddmmyyyy}"
    datetime_format='%Y-%m-%d %H:%M:%S' if config.export_type=="israel" else '%d-%m-%Y %H:%M' # no seconds required
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
    print(f"gauges in train dataset:{train_dataset.dataset.dataset.ims.db.keys()}")
    print(f"links in train dataset:{train_dataset.dataset.dataset.dme.db.keys()}")
    
    print(f'\n\n👀Validation👀')
    validation_dataset = data.create_dataset(
        validation_opt)  # create a validation dataset given opt.dataset_mode and other options
    
    print(f"gauges in validation dataset:{validation_dataset.dataset.dataset.ims.db.keys()}")
    # print(f"norm min-max:{validation_dataset.dataset.dataset.data_dict_B['data_min']}-{validation_dataset.dataset.dataset.data_dict_B['data_max']}")

    print(f"links in validation dataset:{validation_dataset.dataset.dataset.dme.db.keys()}")
    # print(f"norm min-max:{train_dataset.data_dict_B['data_min']}-{train_dataset.data_dict_B['data_max']}")

    model = models.create_model(train_opt)  # create a model given opt.model and other options
    model.setup(train_opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    
    
    for epoch in range(train_opt.n_epochs + train_opt.n_epochs_decay):
        training_losses=defaultdict(float)
#         direction = "AtoB" if (epoch // 10) % 2 == 0 else "BtoA"
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        agg_train_mse_A, agg_train_mse_B, agg_train_bce_B = 0, 0, 0
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

#         print(f"Direction:{direction}")
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += train_opt.batch_size
            epoch_iter += train_opt.batch_size
            
            model.train()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(is_train=True)  # calculate loss functions, get gradients, update network weights
            
            # Training losses
            

            t_comp = (time.time() - iter_start_time) / train_opt.batch_size

            iter_data_time = time.time()
        
            current_losses=model.get_current_losses(is_train=True)
            for key in current_losses:
                training_losses[key] += current_losses[key]
        
        if epoch%100==0 and epoch>0: # TRAIN
            print(f'End of epoch:{epoch}')

            
            
            
        if epoch % ITERS_BETWEEN_VALIDATIONS == 0 and epoch>0: # VALIDATION
            

            print(f"Validation in progress...")
            data_A=validation_dataset.dataset.dataset.dme
            data_B= validation_dataset.dataset.dataset.ims
            k=train_opt.slice_dist
            validation_links=data_A.db_normalized.keys()
            rain_fig, rain_axs = plt.subplots(2, len(validation_links)//2+1, figsize=(15, 15))
            rain_axs=rain_axs.flatten()
            
            
            print(f"Validation links:{validation_links}")
            for link_counter,link in enumerate(validation_links):
                for gauge in validation_link_to_gauge_matching[link]:
                    print(f"with gauge: {gauge}")
                    data_norm_B = data_B.db_normalized[gauge]
                    validation_gauge_full = torch.Tensor(np.array(list(data_norm_B['data'].values())))
    #                 print(f"links:{validation_dataset.dataset.dataset.dme.db_normalized.keys()}")
                    real_fake_gauge_metric[f"{link}-{gauge}"]=0
                    seq_len=0
                    real_gauge_vec=np.array([])
                    fake_gauge_vec=np.array([])
                    fake_gauge_vec_det=np.array([])
                    T=np.array([])

                    # calculate metric for test gauges
                    to_add=0
                    A_delay=0
                    B_delay=0
                    num_samples=len(validation_gauge_full)
                    data_norm_A=data_A.db_normalized[link]
                    validation_link_full=torch.Tensor(np.array(list(data_norm_A['data'].values())))
#                     model.setup(validation_opt,isTrain=False)
                    
                    for batch_counter,i in enumerate(range(0, num_samples, k)): #len(validation_gauge_full)


                        print(f"link:{link}:{i}/{len(validation_link_full)}")
                        print(f"gauge:{gauge}:{i}/{len(validation_gauge_full)}")

                        try:
                            A=validation_link_full[i :  i + k].reshape(k,4)
                            B=validation_gauge_full[i : i + k].reshape(k,1)
                            
                            slice_time=data_norm_B['time'][i: i + k]
                    
                        except RuntimeError:
                            break

                        
                        attenuation_sample=torch.unsqueeze(A.T,0)
                        attenuation_sample_unnormalized = min_max_inv_transform(attenuation_sample,mmin=-50.8,mmax=17)
                        rain_sample=torch.unsqueeze(B.T,0)
                        rain_sample_unnormalized = min_max_inv_transform(rain_sample,mmin=0,mmax=3.3)
                        loader={"link":link, "attenuation_sample":attenuation_sample,\
                         "gague":gauge,\
                         "rain_rate_sample":rain_sample,\
                         "Time":slice_time,\
                         "rain_rate_prob": func_fit(rain_sample_unnormalized,LAMBDA),
                         "distance": torch.tensor([3], device='cuda:0', dtype=torch.float64) # in KM
                        }                      
                        model.set_input(loader,isTrain=False)
                            
                        model.optimize_parameters(is_train=False)
                        visuals = model.get_current_visuals()
                        validation_losses = model.get_current_losses(is_train=False) # validation for each batch, i.e 64 samples


                        if ENABLE_WANDB:
                            # Visualize
                            for key, value in list(validation_losses.items()):
                                validation_losses[f"#{batch_counter}-{key}"] = value
                            wandb.log({**validation_losses})

                            metadata=[0]*4

            
                            with torch.no_grad():
                                visuals = OrderedDict(
                                [
                                    ('real_A', torch.unsqueeze(A.T,0)),\
                                    ('fake_B', model.fake_B_sigmoid),\
                                    ('rec_A', model.rec_A_sigmoid),\
                                    ('real_B',torch.unsqueeze(B.T,0)),\
                                    ('fake_A', model.fake_A_sigmoid),\
                                    ('rec_B', model.rec_B_sigmoid),\
                                    ('fake_B_det_sigmoid', model.fake_B_det_sigmoid),\
                                    ('rec_B_det_sigmoid', model.rec_B_det_sigmoid)
                                ]
                            )
                            
                            real_rain_add=visuals['real_B'][0].cpu().detach().numpy()
                            fake_rain_add=visuals['fake_B'][0].cpu().detach().numpy()
                            fake_detection=model.fake_B_det_sigmoid.cpu().detach().numpy()[0][0]
                            rec_detection=model.rec_B_det_sigmoid.cpu().detach().numpy()[0][0]

                            real_gauge_vec=np.append(real_gauge_vec,np.round(real_rain_add,2))
                            fake_gauge_vec=np.append(fake_gauge_vec,np.round(fake_rain_add,2))
                            fake_gauge_vec_det=np.append(fake_gauge_vec_det, fake_detection)
                            T=np.append(T,np.array(model.t))


                            # rec_A=visuals['rec_A'][0].cpu().detach().numpy()
                            # rec_A_unnorm=min_max_inv_transform(rec_A, mmin=-50.8, mmax=17)
                            # mmin_rec_A=np.min(rec_A_unnorm)
                            # mmax_rec_A=np.max(rec_A_unnorm)

                            # rec_B=visuals['rec_B'][0].cpu().detach().numpy()
                            # rec_B_unnorm=min_max_inv_transform(rec_B, mmin=0, mmax=3.3)
                            # mmin_rec_B=np.min(rec_B_unnorm)
                            # mmax_rec_B=np.max(rec_B_unnorm)


                            fig, axs = plt.subplots(2, 3, figsize=(15, 15))
                            title = f'{batch_counter}:{link}<->{gauge}'
                            
                            for ax, key in zip(axs.flatten(), visuals):
                                N = 4 if 'A' in key else 1

                                # Plot Data
                                data = visuals[key][0].cpu().detach().numpy()
                                assert(data.shape == (N,64))
                                
                            
                                for i in range(4): #This is only validaiton
                                    if 'A' in key:
                                        # Normalization with global values will only work \
                                        # when using Relu or logistic_cdf
                                        # Linear normalization is saturating values
                                        # Relu does not normalize between 0 and 1
                                        mmin = -50.8 #mmin_rec_A if 'fake' in key else -50.8
                                        mmax = 17 #mmax_rec_A if 'fake' in key else 17
                                        label = DME_KEYS[i]
                                        data_vector = torch.tensor(data[i])
                                        
                                    else:
                                        mmin = 0 
                                        mmax = 3.3 #mmax_rec_B if 'fake' in key else 3.3
                                        mmin_B=mmin
                                        mmax_B=mmax
                                        label = IMS_KEYS[0]
                                        data_vector = torch.tensor(data[0])

                                    model_t=model.t
                                    
                                    if key!="fake_B" and key!="rec_B":
                                        ax.plot([mpl_dates.date2num(datetime.strptime(t, datetime_format)) for t in model_t],
                                                min_max_inv_transform(data_vector, mmin=mmin, mmax=mmax),
                                                marker='o',
                                                linestyle='dashed',
                                                linewidth=0.0,
                                                markersize=4,
                                                label=label
                                        )
                                    else:
                                        
                                        if key=="fake_B":
                                            mask=fake_detection
                                        else:
                                            mask=rec_detection
                                        
                                        
                                        mask=(mask >= probability_threshold).astype(int)
                                        cmap=["red" if m else "black" for m in mask]
                                        
                                        ax.scatter([mpl_dates.date2num(datetime.strptime(t, datetime_format)) for t in model_t],
                                                min_max_inv_transform(data_vector, mmin=mmin, mmax=mmax),
                                                marker='o',
                                                linestyle='dashed',
                                                linewidth=0.0,
                                                c=cmap,
                                                label="RainAmout[mm/h]",
                                        )
                                        


                                    ax.set_title(key if train_opt.is_only_dynamic else f'{key} \n'
                                                                                       f' {metadata}', y=0.75, fontdict={'fontsize':6})

                                    
                                ax.xaxis.set_major_formatter(date_format)
                            




                        wandb.log({title: fig})
                        with np.printoptions(threshold=np.inf):
                            print(f"batch #{batch_counter}:{fake_detection}")
                    

                    
                    #Un-normalize back to values in range of real rain
                    real_gauge_vec=min_max_inv_transform(real_gauge_vec, mmin=0, mmax=3.3)
                    fake_gauge_vec=min_max_inv_transform(fake_gauge_vec, mmin=0, mmax=3.3)
                    
                    # Convert continuous values to binary class labels
                    real_gauge_vec_labels = (real_gauge_vec >= threshold).astype(int)
                    fake_gauge_vec_det_labels = ((fake_gauge_vec_det >= probability_threshold)).astype(int)
                    

                    p=Preprocess(link=link,gauge=gauge, epoch=epoch, T=T, real=real_gauge_vec, fake=fake_gauge_vec, detections=fake_gauge_vec_det_labels)
                    fig_preprocessed, axs_preprocessed = plt.subplots(1, 1, figsize=(15, 15))
                    

                    
                    CM=confusion_matrix(real_gauge_vec_labels, p.detections)
                    
                    # Create subplots for given confusion matrices
                    f, axes = plt.subplots(1, 1, figsize=(15, 15))

                    # Plot the first confusion matrix at position (0)
                    axes.set_title("Confusion Mat Detection", size=8)
                    ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=["dry","wet"]).plot(
                        include_values=True, cmap="Blues", ax=axes, colorbar=False, values_format=".0f")

                    # Remove x-axis labels and ticks
                    axes.xaxis.set_ticklabels(['dry', 'wet'])
                    axes.yaxis.set_ticklabels(['dry', 'wet'])

                    
                    wandb.log({f"Confusion Matrix":f})
                    wandb.log({"f1-score Detection": f1_score(p.detections, real_gauge_vec_labels)})
                    wandb.log({"Acc": (CM[0][0]+CM[1][1])/(CM[0][0]+CM[0][1]+CM[1][0]+CM[1][1])})
                    

        


                    preprocessed_time=np.asarray(T) #16436.00694444
                    preprocessed_time_wanb=[mpl_dates.date2num(datetime.strptime(t, datetime_format)) for t in T]
                    
                    
                    axs_preprocessed.plot(preprocessed_time_wanb, p.fake_cumsum, label="CML")
                    axs_preprocessed.plot(preprocessed_time_wanb, p.real_cumsum, "--", label="Gauge")
                    axs_preprocessed.grid()
                    fig_preprocessed.legend()
                    fig_preprocessed.tight_layout()


                    # Specify the number of ticks you want on the x-axis
                    num_ticks = 10

                    # Calculate the step size between ticks
                    step_size = len(preprocessed_time) // num_ticks

                    # Set the ticks on the x-axis
                    axs_preprocessed.set_xticks(preprocessed_time_wanb[::step_size])  # Setting x-ticks
                    axs_preprocessed.set_xticklabels(preprocessed_time_wanb[::step_size], rotation=45)  # Setting x-tick labels with rotation
                    axs_preprocessed.xaxis.set_major_formatter(date_format)

                    wandb.log({f"Virtual (CML) vs Real (Gauge) - {link}-{gauge}":fig_preprocessed})
                    
                    #RMSSE
                    cond=[True if r >= 0.2 or f >= 0.1 else False for r,f in zip(p.real, p.fake)]
                    N=len(p.fake)
                    wandb.log({f"RMSSE-{link}-{gauge}":np.sqrt(np.sum((p.real - p.fake)**2)/N)})
        
                            
                    assert(len(T)==len(real_gauge_vec.flatten()))
                    assert(len(T)==len(fake_gauge_vec.flatten()))
                    

                
            if ENABLE_WANDB:
                for key in current_losses:
                    training_losses[key] = training_losses[key]/(ITERS_BETWEEN_VALIDATIONS*len(train_dataset))
            
                wandb.log({**training_losses})
                path_to_html = f"{v.out_path}/{v.map_name}"
#                 v.draw_cml_map()
#                 wandb.log({"html": wandb.Html(open(path_to_html), inject=False)})
            print(print(f"Validation cycle end..."))

    model.save_networks("latest")