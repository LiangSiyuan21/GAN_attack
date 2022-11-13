from logging import info
from math import cos
# from mmdet_v2200.models import losses
from time import process_time
# from pycparser.c_ast import Label
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import stats as st

from transfer_attack.attack import Attack
# import attack_demo.util as demo_utils


class TIFGSM(Attack):
    r"""
    TIFGSM in the paper 'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks'
    [https://arxiv.org/abs/1904.02884]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        kernel_name (str): kernel name. (DEFAULT: gaussian)
        len_kernel (int): kernel length.  (DEFAULT: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (DEFAULT: 3; see Section 3.2.2 in the paper for explanation)
        eps (float): maximum perturbation. (DEFAULT: 8/255)
        alpha (float): step size. (DEFAULT: 2/255)
        decay (float): momentum factor. (DEFAULT: 0.0)
        steps (int): number of iterations. (DEFAULT: 20)
        resize_rate (float): resize factor used in input diversity. (DEFAULT: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (DEFAULT: 0.5)
        random_start (bool): using random initialization of delta. (DEFAULT: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        # TIFGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.0, random_start=False)
        >>> adv_images = attack(images, labels)
        # M-TIFGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=1.0, resize_rate=0.9, diversity_prob=0.0, random_start=False)
        >>> adv_images = attack(images, labels)
        # TI-DI-FGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)
        # M-TI-DI-FGSM
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, kernel_name='gaussian', len_kernel=15, nsig=3, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False, ub=1, lb=0):
        super(TIFGSM, self).__init__("TIFGSM", model)
        self.eps = eps * (ub-lb)
        self.steps = steps
        self.decay = decay
        self.alpha = alpha * (ub-lb)
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig

        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.ub=ub
        self.lb=lb

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
            
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x


    def forward(self, data, gts):
        r"""
        Overridden.
        """
        # print(len(data))
        if len(data) == 2:
            images = data['img'][0].data[0].clone().detach().to(self.device)
        else:
            images = torch.from_numpy(data).to(self.device)
        # labels = labels.clone().detach().to(self.device)
        # labels = self._transform_label(images, labels)
        
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=self.lb, max=self.ub).detach()
            adv_images_start = torch.clamp(adv_images, min=self.lb, max=self.ub)
        
        new_data = {}
        if len(gts) == 2:
            new_data['img_metas'] = data['img_metas'][0]
        else:
            new_data['img_metas'] = gts[2][0].data[0]
            gt_bboxes = torch.from_numpy(gts[0]).unsqueeze(0)
            gt_labels = torch.from_numpy(gts[1]).unsqueeze(0)

        for i in range(int(self.steps)):
            adv_images.requires_grad = True
            if len(gts) == 2:
                data['img'][0] = adv_images
            
                new_data['img'] = adv_images
                losses = self.model(return_loss=True, **new_data, gt_bboxes=gts[0].unsqueeze(0), gt_labels=gts[1].unsqueeze(0))
            else:
                new_data['img'] = adv_images
            # with torch.no_grad():
            #     result = self.model(return_loss=False, rescale=True, attack_mode=True, **data)
            #     outputs, _ = demo_utils.get_scores_and_labels(result, ncls=80)
            #     outputs = torch.from_numpy(outputs)
                # outputs = result[0][0][:, -1]
                # new_outputs = torch.ones([outputs.size()[0], 80]) * (1-outputs) / 79
                # new_outputs = new_outputs[:, labels] = outputs
                # labels = result[1]
            
            # result = self.model(return_loss=False, rescale=True, attack_mode=True, **data)
                losses = self.model(return_loss=True, **new_data, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
            # outputs, _ = demo_utils.get_scores_and_labels(result, ncls=80)
            # outputs = torch.from_numpy(outputs)
            
            # cost = torch.zeros(1)
            # cost = Variable(cost, requires_grad=True)
            # for output in outputs:
            #     output = output.unsqueeze(0).float().cuda()
            #     for label in labels:
            #         label = label.unsqueeze(0).long().cuda()
            #         temp = result[0][0]
            #         temp2 = result[1]
            # cost = self._targeted*loss(result[0][0], result[1][0]).sum()
            
            # grad = torch.autograd.grad(cost, adv_images, 
                                    #    retain_graph=False, create_graph=False)[0]
            
            self.model.zero_grad()
            if isinstance(losses['loss_cls'], list):
                for i, item in enumerate(losses['loss_cls']):
                    if i == 0:
                        loss_total = item
                    else:
                        loss_total = loss_total+item
                losses['loss_cls'] = loss_total * (-1.0)
            else:    
                losses['loss_cls'] = losses['loss_cls'] * (-1.0)
            losses['loss_cls'].backward()
            grad = adv_images.grad.data

            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding=int((self.len_kernel-1)/2), groups=3)
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.lb, max=self.ub).detach()

        if self.random_start:
            return adv_images_start, adv_images
            
        return adv_images

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
