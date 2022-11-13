#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mga_attack.py
@Time    :   2021/09/13 00:04:10
@Author  :   Siyuan Liang
'''

# here put the import lib
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core.defchararray import not_equal
from blackbox_attack.utils.criterion import loss_fct_with_iou, early_stop_crit_fct_with_iou
from attack_demo.util import visual_iter_process

import numpy as np
import torch
from torch import Tensor as t

from blackbox_attack.black_box_attack import BlackBoxAttack
from blackbox_attack.utils.compute_fcts import lp_step
from attack_demo.util import bbox_to_attack_points, mask_to_attack_points, unique_rows, reppoints_to_attack_points

# import config as flags
# from utils import *
# from utils import image_folder_custom_label
import torch.nn as nn
from transfer_attack.attacks.tifgsm import TIFGSM
# from mi_fgsm import MI_FGSM_ENS


"""
    对代码进行了一些整理，不能指定target类，target攻击是会读取之前保存的target类
"""


class IoUMGAAttack(BlackBoxAttack):
    def __init__(self, max_loss_queries=np.inf, max_crit_queries=np.inf, p='inf', epsilon=8. / 255, lb=0, ub=1, name="mga", attack_model=None, attack_mode=None, attack_logistics=None, loss=None, targeted=False, ori_img=None, model_name=None, zeta=None, lambda1=None, patch_attack=None, keypoints_models=None, 
    pop_size=5, cross_rate=0.7, mutation_rate=0.001, ensemble_models=None, ensemble_iters=10, visual_iters_path=None):
        
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
            keypoints_models=keypoints_models,
            visual_iters_path=visual_iters_path
        )

        # parameters about evolution algorithm
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

        # ensemble MI-FGSM parameters, use ensemble MI-FGSM attack generate adv as initial population
        self.ensemble_models = ensemble_models
        self.iters = ensemble_iters
        self.targeted = targeted
        self.max_queries = ensemble_iters
        self.n_queries = None
        self.generations = self.max_loss_queries # no square patch
        self.best_fitness = None
        self.adv = None
        self.best_delta = None
        self.dones_mask = None
        self.results_records_iter_list = None
        self.stop = False
    

    def genertate_best_img_in_generations(self, delta, xs):
            individual = delta.copy()
            zeros = (individual == 0)
            individual[zeros] = -1

            delta = individual * self.epsilon
            delta = torch.from_numpy(delta).cuda()

            last = torch.from_numpy(xs).cuda() + delta.unsqueeze(0)
            last = torch.clamp(last, self.lb, self.ub)
            last = last.cpu().detach().numpy().transpose([0, 2, 3, 1])

            last = torch.FloatTensor(last.transpose(0,3,1,2))

            return last, self.n_queries

    def over_queries(self, n_queries):
        

        if n_queries[0] in self.results_records_iter_list:
            self.stop = True
        elif n_queries[0] == self.generations:
            self.stop = True
        else:
            self.stop = False

    def fitness_helper(self, individual, x, img_metas, gts, loss_fct):

        # resize to image size
        individual = individual.copy()
        zeros = (individual == 0)
        individual[zeros] = -1

        delta = individual * self.epsilon
        delta = torch.from_numpy(delta).cuda()

        adv = torch.from_numpy(x).cuda() + delta.unsqueeze(0)
        adv = torch.clamp(adv, self.lb, self.ub)
        adv = adv.cpu().detach().numpy().transpose([0, 2, 3, 1])

        # only imagenet dataset needs preprocess
        # logits = predict(self.model, torch2numpy(adv), logits=True)

        # loss = self.loss_fn(logits, y)
        loss = loss_fct(self, adv, img_metas, gts)
        
        self.n_queries = self.n_queries + np.ones(1)

        if self.visual_iters_path is not None:
            self.iters_list.append(int(self.n_queries))
            self.losses_list[0].append(float(loss.item()))
            if self.best_fitness is not None:
                self.losses_list[1].append(float(self.best_fitness))

        over_queries = self.over_queries(self.n_queries)
        # if over_queries is not None:
        #     return over_queries[0], over_queries[1]

        self.dones_mask = early_stop_crit_fct_with_iou(self, adv, img_metas, gts)
        if np.all(self.dones_mask):
            self.adv = torch.FloatTensor(adv.transpose(0,3,1,2))
        # if self.is_success(logits, y):
        #     self.adv = adv.cpu()

        return loss.item()

    def get_fitness(self, lw, xs, img_metas, clean_info, loss_fct):
        first, second = self.idx[0], self.idx[1]
        if self.is_change[first] == 1:
            f1 = self.fitness_helper(lw[0], xs, img_metas, clean_info, loss_fct)
            
            self.pop_fitness[first] = f1
            if f1 > self.best_fitness:
                self.best_fitness = f1
                self.best_delta = lw[0].copy()
            self.is_change[first] = 0

            # self.over_queries(self.n_queries)
            if self.stop == True:
                return None
        else:
            f1 = self.pop_fitness[first]

        if self.is_change[second] == 1:
            f2 = self.fitness_helper(lw[1], xs, img_metas, clean_info, loss_fct)
            
            self.pop_fitness[second] = f2
            if f2 > self.best_fitness:
                self.best_fitness = f2
                self.best_delta = lw[1].copy()
            self.is_change[second] = 0

            # self.over_queries(self.n_queries)
            if self.stop == True:
                return None
        else:
            f2 = self.pop_fitness[second]

        return np.array([f1, f2])

    def cross_over(self, lw):
        cross_point = np.random.rand(self.channel, self.h, self.w) < self.cross_rate
        lw[0, cross_point] = lw[1, cross_point]
        return lw

    def mutate(self, lw):

        # generate mutation point
        mutation_point = np.random.rand(self.channel, self.h, self.w) < self.mutation_rate

        # reverse the value at mutation point 1->0, 0->1
        zeros = (lw[0] == 0)
        ones = (lw[0] == 1)
        lw[0, mutation_point & zeros] = 1
        lw[0, mutation_point & ones] = 0
        return lw

    def init_pop(self, xs, img_metas, clean_info):
        
        deltas = None
        scale_factor = img_metas[0].data[0][0]['scale_factor']
        for n in range(self.pop_size):    
            # attack = TIFGSM(self.ensemble_models, eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=0.0, resize_rate=0.9, diversity_prob=0.0, random_start=True, ub=self.ub, lb=self.lb)
            attack = TIFGSM(self.ensemble_models, eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=True, ub=self.ub, lb=self.lb)
            gt_bboxes = clean_info[0].astype(np.float32)*scale_factor
            gt_labels = clean_info[2].astype(np.int64)
            adv = attack(xs, (gt_bboxes, gt_labels, img_metas))

            # datas, labels = x.repeat((self.pop_size, 1, 1, 1)), y.repeat(self.pop_size)
            # adv = adversary.perturb(datas, labels)

            delta = adv[1].detach().cpu().numpy() - xs
            negative = delta <= 0
            positive = delta > 0
            delta[negative] = 0
            delta[positive] = 1

            if deltas is None:
                deltas = delta
            else:
                deltas = np.append(deltas, delta, axis=0)

        return deltas

    def _suggest(self, xs_t, loss_fct, img_metas, clean_info):
        self.adv = None
        
        if self.n_queries == None:
            self.n_queries = np.zeros(1) # no square
            now_queries = self.n_queries.copy()
        else:
            now_queries = self.n_queries

        # results_records, quires_records, results_records_iter_list = records[0], records[1], records[2]

        xs = self.ori_img.numpy().copy()

        self.batch_size, self.channel, self.h, self.w = xs.shape

        if self.n_queries < self.pop_size:
            if self.n_queries == 0:
                self.pop_fitness = np.ones(self.pop_size) * -4444444
                self.is_change = np.zeros(self.pop_size)
                self.best_fitness = np.zeros(1)

            if self.ensemble_models is None:
                # initial population
                self.pop = np.random.randint(0, 2, (self.pop_size, self.channel, self.h, self.w))
            else:
                self.pop = self.init_pop(xs, img_metas, clean_info)


            # this expense 5 queries, this thy the median always 5
            # init pop fitness, this can reduce query, cause in mga, not all individual changes in a generation
            for n in range(int(self.n_queries), self.pop_size):
                self.pop_fitness[n] = self.fitness_helper(self.pop[n], xs, img_metas, clean_info, loss_fct)

                if self.visual_iters_path is not None:
                    self.best_fitness = np.max(self.pop_fitness)
                
                if self.stop:
                    self.best_delta = self.pop[np.argmax(self.pop_fitness)].copy()
                    last, n_queries = self.genertate_best_img_in_generations(self.best_delta, xs)
                    self.stop = False
                    return last, n_queries - now_queries
                
                if self.adv is not None:
                    return self.adv, self.n_queries
        
            self.best_fitness = np.max(self.pop_fitness)
            self.best_delta = self.pop[np.argmax(self.pop_fitness)].copy()

        while True:
            print('best fitness')
            print(self.best_fitness)          
            self.idx = np.random.choice(np.arange(self.pop_size), size=2, replace=False)
            lw = self.pop[self.idx].copy()  # short for losser winner

            fitness = self.get_fitness(lw, xs, img_metas, clean_info, loss_fct)

            if fitness is None:
                last, n_queries = self.genertate_best_img_in_generations(self.best_delta, xs)
                self.stop = False
                return last, n_queries - now_queries

            # if success, abort early
            if self.adv is not None:
                return self.adv, self.n_queries

            # in target situation, the smaller fitness is, the better
            if self.targeted:
                fidx = np.argsort(-fitness)
            else:
                fidx = np.argsort(fitness)

            lw = lw[fidx]
            lw = self.cross_over(lw)
            lw = self.mutate(lw)

            lw = lw[fidx]
            # update population
            self.pop[self.idx] = lw.copy()

            # losser changed, so fitness also change
            self.is_change[self.idx[fidx[0]]] = 1

        # return None, self.query


