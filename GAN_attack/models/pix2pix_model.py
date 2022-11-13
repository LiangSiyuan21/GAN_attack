from numpy.core.fromnumeric import std
import torch
import torchvision
from .base_model import BaseModel
from . import networks
from mmdet_v2200.core import tensor2imgs
import mmcv


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_Fea_min_L2', 'G_cls_min_cw']
        self.loss_names = opt.loss_names.split(',')
        self.loss_paras = opt.loss_paras.split(',')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A_np']
        self.visual_names = ['real_A_np', 'fake_B_np', 'real_B_np']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def visualize_img(self, name):
        if 'fake' in name:
            img_tensor = getattr(self, name).detach()
        else:
            img_tensor = getattr(self, name)
        # print(img_tensor.size())
        img_metas = self.target_dict["data"]['img_metas'].data[0]
        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)
        imgs = tensor2imgs(img_tensor, mean=mean, std=std)

        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))
            img_show = img_show[...,::-1].copy()
            return img_show

    def set_input(self, input, target_dict):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.target_dict = target_dict
        if 'adv_img' in input.keys():
            self.real_A = input['img' if AtoB else 'adv_img'].data[0].to(self.device)
            self.real_A_np = self.visualize_img('real_A')
            self.real_B = input['adv_img' if AtoB else 'img'].data[0].to(self.device)
            self.real_B_np = self.visualize_img('real_B')  
        else:
            self.real_A = input['img' if AtoB else 'adv_img'].data[0].to(self.device).unsqueeze(0)
            self.real_A_np = self.visualize_img('real_A')
            self.real_B = input['adv_img' if AtoB else 'img'].data[0].to(self.device).unsqueeze(0)
            self.real_B_np = self.visualize_img('real_B')          
        self.max = torch.max(self.real_A)
        self.min = torch.min(self.real_A)
        self.eps = (self.max-self.min)*self.opt.eps

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        ori_w, ori_h = self.real_A.shape[2], self.real_A.shape[3]
        torch_size = torchvision.transforms.Resize((512, 512), interpolation=2)
        perturbation = torch.sign(self.netG(torch_size(self.real_A)))*self.eps
        torch_size = torchvision.transforms.Resize((ori_w, ori_h), interpolation=2)  
        self.fake_B = torch.clip(torch_size(perturbation) + self.real_A, self.min, self.max) # G(A) + A
        self.fake_B_np = self.visualize_img('fake_B')


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        
        for index, name in enumerate(self.loss_names):
            if name == 'G_GAN':
            # First, G(A) should fake the discriminator
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                pred_fake = self.netD(fake_AB)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True) * float(self.loss_paras[index])
                self.loss_G = self.loss_G_GAN

            if name == 'G_L1_fb_rB':
            # Second, G(A) + A = B
                self.loss_G_L1_fb_rB = self.criterionL1(self.fake_B, self.real_B) * float(self.loss_paras[index])
                # print(str(self.loss_G_L1_fb_rB))
                self.loss_G = self.loss_G + self.loss_G_L1_fb_rB.cuda('cuda:0')
            
            if name == 'G_L1_fb_rA':
            # Second, G(A) + A = A
                self.loss_G_L1_fb_rA = self.criterionL1(self.fake_B, self.real_A) * float(self.loss_paras[index])
                # print(str(self.loss_G_L1_fb_rA))
                self.loss_G = self.loss_G + self.loss_G_L1_fb_rA.cuda('cuda:0')
                
            if name == 'G_L2_fb_rB':
                self.loss_G_L2_fb_rB = self.criterionL2(self.fake_B, self.real_B) * float(self.loss_paras[index])
                # print(str(self.loss_G_L2_fb_rB))
                self.loss_G = self.loss_G + self.loss_G_L2_fb_rB.cuda('cuda:0')

            if name == 'G_L2_fb_rA':
                self.loss_G_L2_fb_rA = self.criterionL2(self.fake_B, self.real_A) * float(self.loss_paras[index])
                # print(str(self.loss_G_L2_fb_rA))
                self.loss_G = self.loss_G + self.loss_G_L2_fb_rA.cuda('cuda:0') 

            if name == 'G_Fea_min_Gaussi_L2':    
                # Third, F(G(A)+A) = F(Gaussi)
                # for name, param in self.target_dict['model'].named_parameters():
                #     if param.grad is None:
                #         print(name)
                with torch.no_grad():
                    fake_b_fea = self.target_dict['model'](return_loss=False, rescale=False, attack_mode=True, get_features=True, **{'img_metas':self.target_dict['data']['img_metas'], 'img':[self.fake_B]})
                self.loss_G_Fea_min_Gaussi_L2 = self.criterionL2(fake_b_fea[4], self.target_dict['tar_features'][4]) * float(self.loss_paras[index])
                self.loss_G = self.loss_G + self.loss_G_Fea_min_Gaussi_L2.cuda('cuda:0')

                # self.loss_G_Fea_min_Gaussi_L2 = self.loss_G
            
            if name == 'G_cls_min_cw':
                # Fourth, f(G(A)+A) /= f(A)
                with torch.no_grad():
                    self.loss_G_cls_min_cw = torch.exp(self.target_dict['model'](return_loss=True, **{'img_metas':self.target_dict['data']['img_metas'], 'img':self.fake_B}, gt_bboxes=self.target_dict["gt_bboxes"].unsqueeze(0), gt_labels=self.target_dict["gt_labels"].unsqueeze(0))['loss_cls'] * (-1.0)) * float(self.loss_paras[index])
                # # print(str(self.loss_G_cls_min_cw))
                # self.loss_G_cls_min_cw = self.loss_G
                self.loss_G = self.loss_G + self.loss_G_cls_min_cw.cuda('cuda:0')
        # combine loss and calculate gradients
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A) + A
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        for _ in range(10):
            self.forward()
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
