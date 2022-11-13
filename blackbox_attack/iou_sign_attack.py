# -*- encoding: utf-8 -*-
'''
@File    :   iou_sign_attack.py
@Time    :   2021/01/04 16:31:09
@Author  :   liangsiyuan 
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from blackbox_attack.black_box_attack import BlackBoxAttack
from blackbox_attack.utils.compute_fcts import lp_step, sign, norm
from attack_demo.util import img_to_attack_patch, attack_patch_to_img

class IoUSignAttack(BlackBoxAttack):
    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lb, ub, name, attack_model, attack_mode, loss, targeted, ori_img, model_name, zeta, lambda1, attack_logistics, patch_attack):
        """[Sign Attack for IoU loss and cls loss]

        Args:
            max_loss_queries ([int]): [ maximum number of calls allowed to loss oracle per data pt]
            epsilon ([int]): [radius of lp-ball of perturbation]
            p ([str]): [specifies lp-norm  of perturbation]
            fd_eta ([int]): [forward difference step]
            lb ([float]]): [data lower bound]
            ub ([float]): [data lower bound]
            name ([str]): [name of the attack method]
            attack_model ([model]): [object detection model]
            attack_mode ([bool]): [if True, we will attack the detection model]
            loss ([list]): [optimize object function]
            targeted ([bool]): [if targeted attack, the value is True]
            ori_img ([tensor]): [clean img]
            model_name ([str]): [the name of the attacked model]
            zeta ([float]): [the threshold of the IoU]
            lambda1 ([float]): [the banlance of losses]
            patch_attack([str]): [the way to attacking images for object detection]
        """
        super().__init__(
            max_loss_queries=max_loss_queries,
            epsilon=epsilon,
            p=p,
            lb=lb,
            ub=ub,
            name=name,
            attack_model=attack_model,
            attack_mode=attack_mode,
            loss=loss,
            targeted=targeted,
            ori_img=ori_img,
            model_name=model_name,
            zeta=zeta,
            lambda1=lambda1,
            patch_attack=patch_attack
        )
        # fellow the signHunter     
        self.fd_eta = fd_eta
        self.best_est_deriv = None
        self.xo_t = None
        self.sgn_t = None
        self.h = 0
        self.i = 0

    def _suggest(self, xs_t, loss_fct, img_metas, clean_info):
        if self.patch_attack is None:
            _shape = list(xs_t.shape)
            dim = np.prod(_shape[1:])
        else:
            img_patches_t, _shape, dim = img_to_attack_patch(self.patch_attack, xs_t, clean_info, img_metas)
        add_queries = 0
        if self.is_new_batch:
            self.xo_t = xs_t.clone()
            self.h = 0
            self.i = 0
        if self.i == 0 and self.h == 0:
            if self.patch_attack is None:
                self.sgn_t = sign(torch.ones(_shape[0], dim))
                fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            else:
                self.sgn_t = sign(torch.ones(_shape[0], dim))
                f_img_patches_t = lp_step(img_patches_t, self.sgn_t.view(_shape), self.epsilon, self.p)
                fxs_t = attack_patch_to_img(self.patch_attack, self.xo_t, f_img_patches_t, clean_info, img_metas)
                # img_patches_t = lp_step()
            bxs_t = self.xo_t
            est_deriv = (loss_fct(self, fxs_t.cpu().numpy(), img_metas, clean_info) - loss_fct(self, bxs_t.cpu().numpy(), img_metas, clean_info)) / self.epsilon
            self.best_est_deriv = est_deriv
            add_queries = 3
        chunk_len = np.ceil(dim / (2 ** self.h)).astype(int)
        istart = self.i * chunk_len
        iend = min(dim, (self.i + 1) * chunk_len)
        self.sgn_t[:, istart:iend] *= -1.
        if self.patch_attack is None:
            fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        else:
            f_img_patches_t = lp_step(img_patches_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            fxs_t = attack_patch_to_img(self.patch_attack, self.xo_t, f_img_patches_t, clean_info, img_metas)
        bxs_t = self.xo_t
        est_deriv = (loss_fct(self, fxs_t.cpu().numpy(), img_metas, clean_info) - loss_fct(self, bxs_t.cpu().numpy(), img_metas, clean_info))

        # sign here
        self.sgn_t[[i for i, val in enumerate(est_deriv < self.best_est_deriv) if val], istart: iend] *= -1.

        self.best_est_deriv = (est_deriv >= self.best_est_deriv) * est_deriv + (est_deriv < self.best_est_deriv) * self.best_est_deriv

        # perform the step
        if self.patch_attack is None:
            new_xs = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
        else:
            f_img_patches_t = lp_step(img_patches_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            new_xs = attack_patch_to_img(self.patch_attack, self.xo_t, f_img_patches_t, clean_info, img_metas)

        # update i and h for next iteration
        self.i += 1
        if self.i == 2 ** self.h or iend == dim:
            self.h += 1
            self.i = 0
            # if h is exhausted set xo_t to be xs_t
            if self.h == np.ceil(np.log2(dim)).astype(int) + 1:
                self.xo_t = xs_t.clone()
                self.h = 0
                print('new change')
        
        if self.p == '2':
            norm_diff = norm(new_xs-xs_t)                                                                                                                                                                                       

        if self.patch_attack is None:
            return new_xs, np.ones(_shape[0]) + add_queries
        else:
            return new_xs, np.ones(list(xs_t.shape)[0]) + add_queries    