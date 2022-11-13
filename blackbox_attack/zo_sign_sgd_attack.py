"""
Implements ZO-SIGN-SGD attacks from
"SignSGD via Zeroth-Order Oracle"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch as ch
from torch import Tensor as t
import pdb

from blackbox_attack.black_box_attack import BlackBoxAttack
from blackbox_attack.utils.compute_fcts import lp_step

class ZOSignSGDAttack(BlackBoxAttack):
    """
    ZOSignSGD Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, fd_eta, lr, q, lb, ub, name, attack_model, attack_mode, loss, targeted, ori_img, model_name, zeta, lambda1, attack_logistics):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param q: number of noise samples per NES step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_crit_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         name = name,
                         attack_model=attack_model,
                         attack_mode=attack_mode,
                         loss=loss,
                         targeted=targeted,
                         ori_img=ori_img,
                         model_name=model_name,
                         zeta=zeta,
                         lambda1=lambda1
                         )
        self.q = q
        self.fd_eta = fd_eta
        self.lr = lr

    def _suggest(self, xs_t, loss_fct, img_metas, clean_info):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = ch.zeros_like(xs_t)
        for i in range(self.q):
            # exp_noise = ch.randn_like(xs_t) / (dim ** 0.5)
            # exp_noise = ch.randn_like(xs_t)/10
            exp_noise = ch.randn_like(xs_t)
            fxs_t = xs_t + self.fd_eta * exp_noise
            bxs_t = xs_t
            est_deriv = (loss_fct(self, fxs_t.cpu().numpy(), img_metas, clean_info) - loss_fct(self, bxs_t.cpu().numpy(), img_metas, clean_info)) / self.fd_eta
            gs_t += t(est_deriv.reshape(-1, *[1] * num_axes)) * exp_noise
        gs_t = gs_t/self.q
        # compute the cosine similarity
        # cos_sims, ham_sims = metric_fct(xs_t.cpu().numpy(), gs_t.contiguous().view(_shape[0], -1).cpu().numpy())
        # perform the sign step regardless of the lp-ball constraint
        # this is the main difference in the method.
        # new_xs = lp_step(xs_t, gs_t, self.lr, 'inf')
        new_xs = lp_step(xs_t, gs_t, self.lr, self.p)
        # the number of queries required for forward difference is q (forward sample) + 1 at xs_t
        return new_xs, (self.q + 1) * np.ones(_shape[0])

    def config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_crit_queries": "inf" if np.isinf(self.max_crit_queries) else self.max_crit_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "lr": self.lr,
            "q": self.q,
            "fd_eta": self.fd_eta,
            "attack_name": self.__class__.__name__
        }
