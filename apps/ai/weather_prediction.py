from libs.dl_models.dl_models import DL_Models
import wandb


run = wandb.init()
config = run.config


#pre-train generator and critic
dl_models_obj = DL_Models()
