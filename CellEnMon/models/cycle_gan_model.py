import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from CellEnMon.util.image_pool import SignalPool
from .networks import define_G, define_D, GANLoss
import numpy as np
import os
import config

threshold = float(os.environ["threshold"])
probability_threshold = float(os.environ["probability_threshold"])
THETA=float(os.environ["THETA"])


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)
    return _log_cosh(y_pred - y_true)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning CML-to-Gauge translation without paired data.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A||
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B||
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)  -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.epsilon=1e-12
        self.noise = torch.rand(64, device="cuda:0") * 1.6
        dataset_type_str="Train" if self.isTrain else "Validation"
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['cycle_A', 'G_B', 'cycle_B', 'mse_A', 'mse_B','bce_B','bce_fake_B','bce_rec_B', 'D_A', 'D_B', 'G_A','G_B_only'] #   , 

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B_sigmoid', 'rec_A_sigmoid', "fake_B_det"]
        visual_names_B = ['real_B', 'fake_A_sigmoid', 'rec_B_sigmoid',"fake_B_det_sigmoid","rec_B_det_sigmoid"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = define_G(opt.input_nc_A, opt.output_nc_A, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,direction="AtoB")
        self.netG_B = define_G(opt.input_nc_B, opt.output_nc_B, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, direction="BtoA")

        if self.isTrain:  # define discriminators
            self.netD_A = define_D(opt.input_nc_A, opt.ndf, opt.netD,
                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = define_D(opt.input_nc_B, opt.ndf, opt.netD,
                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = SignalPool(opt.pool_size)  # create signal buffer to store previously generated signals
            self.fake_B_pool = SignalPool(opt.pool_size)  # create signal buffer to store previously generated signals
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss(reduction='none')
            self.criterionIdt = torch.nn.L1Loss()
            self.mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def weight_func(self, x, a):
        return 1/(a * torch.exp(-x*a))

    def set_input(self, input, isTrain=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) if isTrain else input["attenuation_sample" if AtoB else 'rain_rate_sample'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device) if isTrain else input['rain_rate_sample' if AtoB else 'attenuation_sample'].to(self.device)
        self.gague = input['gague']
        self.link = input['link']
        self.t = input['Time']
        self.dataset_type="Train" if isTrain else "Validation"
        self.isTrain=isTrain
        self.rain_rate_prob = input['rain_rate_prob'].to(self.device)
        L=input['distance'].to(self.device)
        self.L=L+self.epsilon
        self.dist_func=1/(torch.log(1+(self.L/config.TRAIN_RADIUS)))
        
        if isTrain:
            self.alpha=0.02
            self.metadata_A = input['metadata_A' if AtoB else 'metadata_B'].to(self.device)
            self.metadata_B = input['metadata_B' if AtoB else 'metadata_A'].to(self.device)
            self.attenuation_prob = input['attenuation_prob'].to(self.device)

            
            self.link_norm_metadata=input['link_norm_metadata']
            self.link_metadata=input['link_metadata']
            self.link_full_name=input['link_full_name'][0]
            self.link_center_metadata=input['link_center_metadata']


            self.gague_norm_metadata=input['gague_norm_metadata']
            self.gague_metadata=input['gague_metadata']
            self.gague_full_name=input['gague_full_name'][0]

            self.data_transformation = input['data_transformation']
            self.metadata_transformation = input['metadata_transformation']

    def dynamic_norm_zero_one(self,x, db_type): #
        epsilon=1e-6
        min_val = torch.min(x)
        max_val = torch.max(x)

        global_min = -50.8 if db_type=="dme" else 0
        global_max = 17 if db_type=="dme" else 3.3

        
        # Clipping min_val and max_val
        min_val = min_val if min_val < global_min else global_min
        max_val = max_val if max_val > global_max else global_max

        return (x - min_val) / (max_val - min_val + epsilon)

    def norm_zero_one(self,x):
        epsilon=1e-6
        min_val = torch.min(x)
        max_val = torch.max(x)

        return (x - min_val) / (max_val - min_val + epsilon)


    def logistic_cdf(self, x):
        return 1 / (1 + torch.exp(-x / 5))

    def norm_mean_std(self,x):
        return (x-x.mean())/(x.std()+self.epsilon)

        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""          
        ## >> B
        fake_B = self.netG_A(self.real_A, dir="AtoB")   # G_A(A)

        activation=nn.LeakyReLU()
        
        self.fake_B_det = self.norm_mean_std(fake_B[1])
        self.fake_B_det_sigmoid = torch.sigmoid(self.fake_B_det) ## <-- detection

        # print(self.L)

        ## >> Fake
            # >> A
        self.fake_A = self.netG_B(self.real_B,dir="BtoA") 
        self.fake_A_sigmoid=self.fake_A ## <<-- regression

            # >> B
        self.fake_B=fake_B[0]
        
        self.fake_B_sigmoid = activation(self.fake_B) ##self.norm_mean_std() <<-- regression
        self.fake_B_with_detection = self.fake_B_sigmoid * (self.fake_B_det_sigmoid > probability_threshold)

        

        

        ## >> Rec
            # >> A
            ## <--what if detector is wrong?? we need a way to propogate this to regression
        self.rec_A = self.netG_B(self.fake_B_sigmoid, dir="BtoA") 
        self.rec_A_sigmoid=self.rec_A ## <<-- regression

            # >> B
        rec_B=self.netG_A(self.fake_A_sigmoid,dir="AtoB")


        self.rec_B_det=self.norm_mean_std(rec_B[1])
        self.rec_B_det_sigmoid=torch.sigmoid(self.rec_B_det) ## <-- detection
        
        
        self.rec_B=rec_B[0]
        self.rec_B_sigmoid = activation(self.rec_B) ## self.norm_mean_std() <<-- regression
        self.rec_B_with_detection = self.rec_B_sigmoid * (self.rec_B_det_sigmoid > probability_threshold)

        
        

    def backward_D_basic(self, netD, real, fake): #weight=torch.ones([1], device='cuda:0')
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake)
        if self.isTrain:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        #fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, self.fake_A_sigmoid)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        #fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, self.fake_B_sigmoid)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = 10
        lambda_B = 10
        # Identity loss
        if lambda_idt > 0:

            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B,dir="BtoA")
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A, dir="AtoB")[0]
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.rr_norm = self.rain_rate_prob
        self.att_norm = self.alpha + 1 - self.attenuation_prob
        const=81.66 if self.dataset_type=="Train" else 1 #33.44
        pos_weight = torch.tensor([const], dtype=torch.float32, device="cuda:0") # wet event is x times more important (!!!)
        
        targets=(self.real_B >= threshold/3.3).float() # 0.2/3.3=0.06, ie. we consider a wet event over 
        #adjust for type-1 and type-2 erros
        adjusted_weights=self.rr_norm.clone()
        adjusted_fake_weights=self.weight_func(torch.abs(self.real_B-self.fake_B_sigmoid),THETA)
        adjusted_fake_weights_det=self.weight_func(torch.abs(targets-self.fake_B_det_sigmoid),THETA)
        #---
        adjusted_rec_weights=self.weight_func(torch.abs(self.real_B-self.rec_B_sigmoid),THETA)
        adjusted_rec_weights_det=self.weight_func(torch.abs(targets-self.rec_B_det_sigmoid),THETA)

        


        
        # adjusted_fake_weights[(self.fake_B_det_sigmoid < probability_threshold) & (targets==1)] = 100
        # adjusted_fake_weights[(self.fake_B_det_sigmoid > probability_threshold) & (targets==0)] = 100
        fake_bce_weight_loss = nn.BCELoss(reduction="none") #weight=self.rain_rate_prob #nn.BCEWithLogitsLoss(pos_weight=self.rain_rate_prob) # 
        fake_focal_loss = FocalLoss()
        # adjusted_rec_weights[(self.rec_B_det_sigmoid < probability_threshold) & (targets==1)] = 100
        # adjusted_rec_weights[(self.rec_B_det_sigmoid > probability_threshold) & (targets==0)] = 100
        rec_bce_weight_loss = nn.BCELoss(reduction="none") #weight=self.rain_rate_prob #nn.BCEWithLogitsLoss(pos_weight=self.rain_rate_prob) # 
        rec_focal_loss = FocalLoss()


        

        
        # works best **without** weights: self.rain_rate_prob
        self.loss_bce_fake_B = torch.sum(fake_focal_loss(self.fake_B_det , targets)) 
        self.loss_bce_rec_B  = torch.sum(rec_focal_loss(self.rec_B_det, targets))
        self.loss_bce_B = self.loss_bce_fake_B + self.loss_bce_rec_B
        
        ## <--what if detector is wrong?? we need a way to bring down high values
        self.D_B=self.netD_B(self.fake_B_sigmoid) # + self.noise
        self.loss_G_B_only=self.criterionGAN(self.D_B, True) # weight=self.rr_norm.max()

        # GAN loss D_B(G_A(A))
        self.loss_G_B =  self.loss_bce_B + self.loss_G_B_only

        # GAN loss D_A(G_B(B))
        self.D_A=self.netD_A(self.fake_A_sigmoid)
        self.loss_G_A = self.criterionGAN(self.D_A, True) #weight=self.rr_norm.max(), weight=self.att_norm.mean()
        


        if "DEBUG" in os.environ and int(os.environ["DEBUG"]):
            print(f"real A:{self.real_A.shape}")
            print(f"real B:{self.real_B.shape}")
            print(f"fake A:{self.fake_A.shape}")
            print(f"fake B:{self.fake_B.shape}")
            print(f"rec A: {self.rec_A.shape}")
            print(f"rec B:{self.rec_B.shape}")
            print(f"D(fake_B): {self.netD_B(self.fake_B).shape}")
            print(f"D(fake_A): {self.netD_A(self.fake_A).shape}")

            print(f"self.fake_B_det_sigmoid:{self.fake_B_det_sigmoid}")
            print(f"rr_prob: {self.rain_rate_prob.shape}")
            print(f"fake_B: {self.fake_B.shape}")
            print(f"fake_A: {self.fake_A.shape}")

            print(f"fake_B * rr_prob: {(self.fake_B * self.rain_rate_prob).shape}")
            print(f"rec_B * rr_prob: {(self.rec_B * self.rain_rate_prob).shape}")
        
        
        # print(self.L)

        LogCosh=LogCoshLoss()
        L1=nn.L1Loss(reduction='none') # weight=self.rr_norm
        L2=nn.MSELoss(reduction='none') # weight=self.rr_norm
        # adjusted_rec_weights[(self.fake_B_det_sigmoid <= probability_threshold ) & (targets=1)] = some_large_number

        rec_A_unnorm=self.min_max_inv_transform(self.rec_A_sigmoid,-50.8,17)
        real_A_unnorm=self.min_max_inv_transform(self.real_A,-50.8,17)

        self.loss_cycle_A = torch.mean(L1(rec_A_unnorm, real_A_unnorm)) #* self.att_norm
                                       
        # Backward cycle loss || G_A(G_B(B)) - B|| # self.rain_rate_prob 
        ## <--what if detector is wrong?? we need a way to bring down high values
        rec_B_unnorm=self.min_max_inv_transform(self.rec_B_sigmoid, 0, 3.3)
        
        real_B_unnorm=self.min_max_inv_transform(self.real_B, 0, 3.3)
        
        
        # --> torch.sum is REALLY important here.
        # --> Remember most of dataset does not have rain events, so we don't need to include this in the loss
        # --> and rain events, or mistakes need to be punished harshly!
        # --> SOFT: LAMBDA=0.27
        self.loss_cycle_B = torch.sum(L1(rec_B_unnorm, real_B_unnorm) * self.rain_rate_prob)
        # self.loss_cycle_B = RMSLE(rec_B_unnorm,real_B_unnorm)

        self.loss_mse_A = torch.sum(self.criterionCycle(self.fake_A_sigmoid, self.real_A))
        self.loss_mse_B = torch.sum(self.criterionCycle(self.fake_B_sigmoid, self.real_B))

        # combined loss and calculate gradients
        # cycle_A and cycle_B should be the same scale - mind the training/validation losses (!!!)
        self.loss_G =\
            (     
                10 * self.loss_cycle_B +\
                self.loss_cycle_A +\

                # self.loss_bce_fake_B+\
                # self.loss_bce_rec_B+\

                self.loss_G_B_only +\
                self.loss_G_A

            )

            #*self.dist_func *

            



            

             
        

        if self.isTrain:
            self.loss_G.backward()

   
    
    def min_max_inv_transform(self,x, mmin, mmax):
        return x # x * (mmax - mmin) + mmin
    
    def optimize_parameters(self, is_train=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        
        dataset_type_str="Train" if is_train else "Validation"
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        
        
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        if self.isTrain:
            self.optimizer_G.step()  # update G_A and G_B's weights
        
        # D_A and D_B
        ## resetting attrs ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'mse_A', 'mse_B', 'bce_B','G_B_only']

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero        
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        if self.isTrain:
            self.optimizer_D.step()  # update D_A and D_B's weights


        
        setattr(self,f"loss_{self.dataset_type}_D_A",self.loss_D_A)
        setattr(self,f"loss_{self.dataset_type}_G_A",self.loss_G_A)
        setattr(self,f"loss_{self.dataset_type}_D_B",self.loss_D_B)

        setattr(self,f"loss_{self.dataset_type}_cycle_A",self.loss_cycle_A)
        setattr(self,f"loss_{self.dataset_type}_G_B",self.loss_G_B)
        setattr(self,f"loss_{self.dataset_type}_cycle_B",self.loss_cycle_B)
        setattr(self,f"loss_{self.dataset_type}_mse_A",self.loss_mse_A)
        setattr(self,f"loss_{self.dataset_type}_mse_B",self.loss_mse_B)
        setattr(self,f"loss_{self.dataset_type}_G_B_only",self.loss_G_B_only)
        setattr(self,f"loss_{self.dataset_type}_bce_B",self.loss_bce_B)
        setattr(self,f"loss_{self.dataset_type}_bce_fake_B",self.loss_bce_fake_B)
        setattr(self,f"loss_{self.dataset_type}_bce_rec_B",self.loss_bce_rec_B)
