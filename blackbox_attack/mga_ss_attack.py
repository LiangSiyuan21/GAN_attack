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
import time

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

class MGASSAttack(BlackBoxAttack):
    def __init__(self, max_loss_queries, epsilon, p, p_init, lb, ub, name, attack_model, attack_mode, attack_logistics, loss, targeted, ori_img, model_name, zeta, lambda1, patch_attack, keypoints_models, pop_size, cross_rate, mutation_rate, ensemble_models, ensemble_iters, visual_iters_path, divide_conquer_num, divide_conquer_restart, fitness_style, attack_parallel,mga_iters, loss_log, mix_init, gaussian):
    
    # max_loss_queries=..., max_crit_queries=..., epsilon=0.5, p='inf', p_init=0.05, lb=0, ub=1, name="nes", attack_model=None, attack_mode=None, attack_logistics=None, loss=None, targeted=None, ori_img=None, model_name=None, zeta=None, lambda1=None, patch_attack=None, keypoints_models=None, pop_size=5, cross_rate=0.7, mutation_rate=0.001, ensemble_models=None, ensemble_iters=10, visual_iters_path=None, divide_conquer_num=1, divide_conquer_restart=False, fitness_style=None, attack_parallel=None):
        # super().__init__(max_loss_queries, max_crit_queries, epsilon, p, p_init, lb, ub, name, attack_model, attack_mode, attack_logistics, loss, targeted, ori_img, model_name, zeta, lambda1, patch_attack, keypoints_models, pop_size, cross_rate, mutation_rate, ensemble_models, ensemble_iters, visual_iters_path, divide_conquer_num, divide_conquer_restart, fitness_style, attack_parallel)        
        super().__init__(
            max_loss_queries=max_loss_queries, 
            epsilon=epsilon, 
            p=p, 
            lb=lb, 
            ub=ub, 
            name=name,
            attack_model=attack_model,
            attack_mode=attack_mode,
            attack_logistics=attack_logistics,
            loss=loss,
            targeted=targeted,
            ori_img=ori_img,
            model_name=model_name,
            zeta=zeta,
            lambda1=lambda1,
            patch_attack=patch_attack,
            keypoints_models=keypoints_models,
            pop_size=pop_size,
            cross_rate=cross_rate,
            mutation_rate=mutation_rate,
            ensemble_models=ensemble_models,
            ensemble_iters=ensemble_iters,
            visual_iters_path=visual_iters_path,
            gaussian=gaussian
            )

        # parameters about evolution algorithm
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate


        #parameters about square attack
        self.i = 0
        self.attack_parallel = attack_parallel
        self.p_init = p_init
        self.p = p
        self.mga_flag = None
        self.p_change = None
        if self.patch_attack is not None:
            self.attack_points = None
        
        # paremeters about divide and conquer
        self.divide_conquer_num = int(divide_conquer_num)
        self.divide_conquer_restart = divide_conquer_restart
        self.fitness_style = fitness_style
        self.mga_iters = mga_iters
        self.loss_log = loss_log
        self.mix_init = mix_init


        # ensemble MI-FGSM parameters, use ensemble MI-FGSM attack generate adv as initial population
        self.ensemble_models = ensemble_models
        self.iters = ensemble_iters
        self.targeted = targeted
        self.max_queries = ensemble_iters
        self.n_queries = np.zeros(1)
        self.generations = self.max_loss_queries # no square patch
        self.best_fitness = None
        self.adv = None
        self.best_delta = None
        self.dones_mask = None
        self.results_records_iter_list = None
        self.stop = False

    def init_pop(self, xs, img_metas, clean_info, attack_type=None):
        
        deltas = None
        scale_factor = img_metas[0].data[0][0]['scale_factor']
        if attack_type:
            for n in range(1):
                if len(self.ensemble_models) > 1:
                    if attack_type == 'ssd_mdtifgsm':
                        attack = TIFGSM(self.ensemble_models[1], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'ssd_dtifgsm':
                        attack = TIFGSM(self.ensemble_models[1], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=0.0, resize_rate=0.9, diversity_prob=0.7, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'ssd_mtifgsm':
                        attack = TIFGSM(self.ensemble_models[1], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=1.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'ssd_tifgsm':
                        attack = TIFGSM(self.ensemble_models[1], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=0.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'mdtifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'dtifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=0.0, resize_rate=0.9, diversity_prob=0.7, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'mtifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=1.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'tifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=0.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=self.ub, lb=self.lb)
                else:
                    if attack_type == 'mdtifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'dtifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=0.0, resize_rate=0.9, diversity_prob=0.7, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'mtifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=1.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=self.ub, lb=self.lb)
                    elif attack_type == 'tifgsm':
                        attack = TIFGSM(self.ensemble_models[0], eps=12.5/255, alpha=3/255, steps=self.max_queries, decay=0.0, resize_rate=0.9, diversity_prob=0.0, random_start=False, ub=self.ub, lb=self.lb)
                gt_bboxes = clean_info[0].astype(np.float32)*scale_factor
                gt_labels = clean_info[2].astype(np.int64)
                adv = attack(xs, (gt_bboxes, gt_labels, img_metas))

                # datas, labels = x.repeat((self.pop_size, 1, 1, 1)), y.repeat(self.pop_size)
                # adv = adversary.perturb(datas, labels)

                delta = adv.detach().cpu().numpy() - xs
                negative = delta <= 0
                positive = delta > 0
                delta[negative] = 0
                delta[positive] = 1

                individual = delta.copy()
                zeros = (individual == 0)
                individual[zeros] = -1

                delta = individual * self.epsilon

                if deltas is None:
                    deltas = delta
                else:
                    deltas = np.append(deltas, delta, axis=0)
        else:
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

            individual = delta.copy()
            zeros = (individual == 0)
            individual[zeros] = -1

            delta = individual * self.epsilon

            if deltas is None:
                deltas = delta
            else:
                deltas = np.append(deltas, delta, axis=0)

        return deltas

    def over_queries(self, n_queries):
        

        if n_queries[0] in self.results_records_iter_list:
            self.stop = True
        elif n_queries[0] == self.generations:
            self.stop = True
        else:
            self.stop = False

    def genertate_best_img_in_generations(self, delta, xs):

            delta = torch.from_numpy(delta).cuda()

            last = torch.from_numpy(xs).cuda() + delta.unsqueeze(0)
            last = torch.clamp(last, self.lb, self.ub)
            last = last.cpu().detach().numpy().transpose([0, 2, 3, 1])

            last = torch.FloatTensor(last.transpose(0,3,1,2))

            return last, self.n_queries
    
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

    def get_child_fitness(self, lw, xs, img_metas, clean_info, loss_fct, child_s, child_windows):
        first, second = self.idx[0], self.idx[1]
        if self.is_change[first] == 1:
            f1_pop_fitness, f1_child_fitness= self.fitness_child_helper(lw[0], xs[0], img_metas, clean_info, loss_fct, child_s, child_windows, self.fitness_style)
            self.pop_fitness[first] = f1_pop_fitness
            self.pop_children_fitness[:, first] = f1_child_fitness
            if f1_pop_fitness > self.best_fitness:
                self.best_fitness = f1_pop_fitness
                self.best_delta = lw[0].copy()
            self.is_change[first] = 0
        
            # if self.stop == True:
            #     return None
        else:
            f1_pop_fitness = self.pop_fitness[first]
            f1_child_fitness = self.pop_children_fitness[:, first]

        if self.is_change[second] == 1:
            f2_pop_fitness, f2_child_fitness = self.fitness_child_helper(lw[1], xs[1], img_metas, clean_info, loss_fct, child_s, child_windows, self.fitness_style)
            self.pop_fitness[second] = f2_pop_fitness
            self.pop_children_fitness[:, second] = f2_child_fitness
            if f2_pop_fitness > self.best_fitness:
                self.best_fitness = f2_pop_fitness
                self.best_delta = lw[1].copy()
            self.is_change[second] = 0

            # self.over_queries(self.n_queries)
            # if self.stop == True:
            #     return None
        else:
            f2_pop_fitness = self.pop_fitness[second]
            f2_child_fitness = self.pop_children_fitness[:, second]

        return np.array([f1_pop_fitness, f2_pop_fitness], dtype=object), np.stack([f1_child_fitness, f2_child_fitness])
    
    def fitness_child_helper(self, individual, x, img_metas, gts, loss_fct, child_s, child_windows, fitness_style):

        delta = torch.from_numpy(individual).cuda()

        adv = torch.from_numpy(x).cuda() + delta.unsqueeze(0)
        adv = torch.clamp(adv, self.lb, self.ub)
        adv = adv.cpu().detach().numpy().transpose([0, 2, 3, 1])

        loss, loss_children = loss_fct(self, adv, img_metas, gts, child_s, child_windows, fitness_style)
        self.n_queries = self.n_queries + np.ones(1)

        # if self.visual_iters_path is not None:
        #     self.iters_list.append(int(self.n_queries))
        #     self.losses_list[0].append(float(loss.item()))
        #     if self.best_fitness is not None:
        #         self.losses_list[1].append(float(self.best_fitness))

        # self.over_queries(self.n_queries)

        # self.dones_mask = early_stop_crit_fct_with_iou(self, adv, img_metas, gts)
        # if np.all(self.dones_mask):
        #     self.adv = torch.FloatTensor(adv.transpose(0,3,1,2))

        
        
        return loss[0], loss_children

    def cross_over(self, lw, child_windows, child_step, child_fitness):
        for n in range(len(child_windows)):
            cross_point = np.random.rand(3, child_step, child_step) < self.cross_rate
            child_h, child_w = int(child_windows[n][0]), int(child_windows[n][1])
            if (child_fitness[0][n] == 22222.0) and (child_fitness[1][n] == 22222.0):
                continue
            elif (child_fitness[0][n] < child_fitness[1][n]):
                lw[0, :, child_h:child_h+child_step, child_w:child_w+child_step][cross_point] = lw[1, :, child_h:child_h+child_step, child_w:child_w+child_step][cross_point].copy()
                self.is_change_flag[0] = 1
            elif (child_fitness[1][n] < child_fitness[0][n]):
                lw[1, :, child_h:child_h+child_step, child_w:child_w+child_step][cross_point] = lw[0, :, child_h:child_h+child_step, child_w:child_w+child_step][cross_point].copy()
                self.is_change_flag[1] = 1
        return lw
    
    def mutate(self, lw, child_windows, child_step, child_fitness):
        # generate mutation point
        # reverse the value at mutation point 1->0, 0->1
        for n in range(len(child_windows)):
            mutation_point = np.random.rand(3, child_step, child_step) < self.mutation_rate
            child_h, child_w = int(child_windows[n][0]), int(child_windows[n][1])
            if (child_fitness[0][n] == 22222.0) and (child_fitness[1][n] == 22222.0):
                continue
            elif (child_fitness[0][n] < child_fitness[1][n]):
                zeros = (lw[0, :, child_h:child_h+child_step, child_w:child_w+child_step] == -self.epsilon)
                ones = (lw[0, :, child_h:child_h+child_step, child_w:child_w+child_step] == self.epsilon)
                lw[0, :, child_h:child_h+child_step, child_w:child_w+child_step][mutation_point & zeros] = self.epsilon
                lw[0, :, child_h:child_h+child_step, child_w:child_w+child_step][mutation_point & ones] = -self.epsilon
            elif (child_fitness[1][n] < child_fitness[0][n]):
                zeros = (lw[1, :, child_h:child_h+child_step, child_w:child_w+child_step] == -self.epsilon)
                ones = (lw[1, :, child_h:child_h+child_step, child_w:child_w+child_step] == self.epsilon)
                lw[1, :, child_h:child_h+child_step, child_w:child_w+child_step][mutation_point & zeros] = self.epsilon
                lw[1, :, child_h:child_h+child_step, child_w:child_w+child_step][mutation_point & ones] = -self.epsilon
        return lw

    def fitness_helper(self, individual, x, img_metas, gts, loss_fct):
    
        # resize to image size
        delta = torch.from_numpy(individual).cuda()

        adv = torch.from_numpy(x).cuda() + delta.unsqueeze(0)
        adv = torch.clamp(adv, self.lb, self.ub)
        adv = adv.cpu().detach().numpy().transpose([0, 2, 3, 1])

        # only imagenet dataset needs preprocess
        # logits = predict(self.model, torch2numpy(adv), logits=True)

        # loss = self.loss_fn(logits, y)
        loss = loss_fct(self, adv, img_metas, gts)
        self.n_queries = self.n_queries + np.ones(1)

        
        self.best_fitness = 0.0
        if self.visual_iters_path is not None:
            self.iters_list.append(int(self.n_queries))
            self.losses_list[0].append(float(loss.item()))
            if self.best_fitness is not None:
                self.losses_list[1].append(float(self.best_fitness))

        self.over_queries(self.n_queries)

        self.dones_mask = early_stop_crit_fct_with_iou(self, adv, img_metas, gts)
        if np.all(self.dones_mask):
            self.adv = torch.FloatTensor(adv.transpose(0,3,1,2))
        # if self.is_success(logits, y):
        #     self.adv = adv.cpu()

        return loss.item()

    def attack_points_selection(self, patch_attacks, it, h, w, clean_info, img_metas, s, ori_img):
        # print('it is {}'.format(it))
        attack_points = np.empty(shape=[0, 2])
        if it in [0, 11, 51, 201, 501, 1001, 2001, 4001, 6001, 8001, 10001]:
            for patch_attack in patch_attacks.split(','):
                if patch_attack == 'bbox':
                    attack_points_bbox = bbox_to_attack_points(h, w, clean_info, img_metas, s)
                    if attack_points_bbox.shape[0]>0:
                        attack_points = np.append(attack_points, attack_points_bbox, axis=0)
                    else:
                        attack_points = attack_points
                elif patch_attack == 'maskrcnn':
                    data = {}
                    data['img'] = [ori_img]
                    data['img_metas'] = img_metas
                    with torch.no_grad():
                        results = self.keypoints_models[patch_attack](return_loss=False, rescale=True, **data)
                    seg_masks = results[0][1]
                    attack_points_mask = mask_to_attack_points(clean_info, seg_masks, img_metas, s=s)
                    attack_points = np.append(attack_points, attack_points_mask, axis=0)
                elif patch_attack == 'reppoints':
                    data = {}
                    data['img'] = [ori_img]
                    data['img_metas'] = img_metas
                    with torch.no_grad():
                        results = self.keypoints_models[patch_attack](return_loss=False, rescale=True, **data, get_points=True)
                    attack_points_rep = reppoints_to_attack_points(results, h=h, w=w, s=s)
                    attack_points = np.append(attack_points, attack_points_rep, axis=0)
                elif patch_attack == 'proposal':
                    data = {}
                    data['img'] = [ori_img]
                    data['img_metas'] = img_metas
                    with torch.no_grad():
                        results = self.keypoints_models[patch_attack](return_loss=False, rescale=True, **data, get_proposal=True)
                    proposals = results[0].cpu().detach().numpy()[:, :4]
                    attack_points_bbox = bbox_to_attack_points(h, w, proposals, img_metas, s, get_proposal=True)
                    attack_points = np.append(attack_points, attack_points_bbox, axis=0)  
            attack_points_unique = unique_rows(attack_points)
            self.attack_points = np.array(attack_points_unique, dtype='int64')
        return self.attack_points
    

    def attack_parallel_selection(self, parallel_num_init, it, n_iters):
        # reverse
        parallel_num_init = int(parallel_num_init)
        a = 1
        # if it <= 20:
        #     parallel_num = parallel_num_init * 4
        # if 20 < it <= 100:
        #     parallel_num = parallel_num_init * 4
        # elif 100 < it <= 400:
        #     parallel_num = parallel_num_init * 2
        # elif 400 < it <= 1000:
        #     parallel_num = parallel_num_init * 2
        # elif 1000 < it <= 2000:
        #     parallel_num = parallel_num_init * 1
        # elif 2000 < it <= 4000:
        #     parallel_num = parallel_num_init * 1
        # elif 4000 < it <= 8000:
        #     parallel_num = parallel_num_init * 1
        # else:
        #     parallel_num = parallel_num_init

        if it <= 20 * a:
            parallel_num = parallel_num_init * 4
        if 20 *a < it <= 100 *a:
            parallel_num = parallel_num_init * 4
        elif 100*a < it <= 400 *a:
            parallel_num = parallel_num_init * 2
        elif 400*a < it <= 1000*a:
            parallel_num = parallel_num_init * 2
        elif 1000*a < it <= 2000*a:
            parallel_num = parallel_num_init * 1
        elif 2000*a < it <= 4000*a:
            parallel_num = parallel_num_init * 1
        elif 4000*a < it <= 8000*a:
            parallel_num = parallel_num_init * 1
        else:
            parallel_num = parallel_num_init

        if it in [0, 21, 101, 401, 1001, 2001, 4001]:
            flag = True
        else:
            flag = False

        # if it <= 2:
        #     parallel_num = parallel_num_init * 4
        # if 2 < it <= 10:
        #     parallel_num = parallel_num_init * 4
        # elif 10 < it <= 40:
        #     parallel_num = parallel_num_init * 2
        # elif 40 < it <= 100:
        #     parallel_num = parallel_num_init * 2
        # elif 100 < it <= 200:
        #     parallel_num = parallel_num_init * 1
        # elif 200 < it <= 400:
        #     parallel_num = parallel_num_init * 1
        # elif 400 < it <= 800:
        #     parallel_num = parallel_num_init * 1
        # else:
        #     parallel_num = parallel_num_init

        # if it in [0, 3, 11, 41, 101, 201, 401]:
        #     flag = True
        # else:
        #     flag = False

        return parallel_num, flag

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 20 < it <= 100:
            p = p_init / 2
        elif 100 < it <= 400:
            p = p_init / 4
        elif 400 < it <= 1000:
            p = p_init / 8
        elif 1000 < it <= 2000:
            p = p_init / 16
        elif 2000 < it <= 4000:
            p = p_init / 32
        elif 4000 < it <= 8000:
            p = p_init / 64
        else:
            p = p_init

        return p

    def _suggest(self, xs_t, loss_fct, img_metas, clean_info):
        xs = xs_t.cpu().numpy().transpose(0,3,1,2)
        if xs.shape[0] != self.pop_size:
            xs = xs.repeat(self.pop_size, axis=0)
        else:
            xs = xs
        c, h, w = xs.shape[1:]
        n_features = c*h*w
        now_queries = self.n_queries.copy()
        p = self.p_selection(self.p_init, self.n_queries, 10000)
        if self.p == 'inf':
            if self.is_new_batch:
                if self.visual_iters_path is not None:
                    self.iters_list.append(0)
                    self.losses_list[0].append(0)
                    self.losses_list[1].append(0)   

                self.p_change = p
                self.x = xs.copy()
                self.pop = np.ones(xs.shape)

                self.pop_fitness = np.ones(self.pop_size)*-44444.0
                if self.ensemble_models is None:
                    # initial population
                    self.pop = np.random.choice([-self.epsilon, self.epsilon], size=[1, c, 1, w])
                else:
                    if self.mix_init:
                        attack_list = self.mix_init.split(',')
                        for idx in range(0, self.pop_size):
                            if attack_list[idx] == "vertical":
                                self.pop[idx] = np.random.choice([-self.epsilon, self.epsilon], size=[1, c, 1, w])
                                continue
                            self.pop[idx] = self.init_pop(xs[idx][np.newaxis, :, :, :], img_metas, clean_info, attack_type=attack_list[idx])
                    else:
                        for idx in range(0, self.pop_size):
                            self.pop[idx] = self.init_pop(xs[idx][np.newaxis, :, :, :], img_metas, clean_info)
                
                xs = np.clip(xs+self.pop, self.lb, self.ub)
                for n in range(self.pop_size):
                    self.pop_fitness[n] = loss_fct(self, xs[n][np.newaxis, :, :, :].transpose(0,2,3,1), img_metas, clean_info)  
                    self.n_queries += 1
                self.best_fitness = np.max(self.pop_fitness)
                self.mga_flag = False
            
            deltas = xs - self.x
            p = self.p_selection(self.p_init, self.n_queries, 10000)
            if self.p_change != p:
                self.mga_flag = False
                self.p_change = p
            
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            child_step = int(s/self.divide_conquer_num)

            if self.patch_attack is not None:
                attack_points = self.attack_points_selection(self.patch_attack, self.n_queries, h, w, clean_info, img_metas, s=s, ori_img=self.ori_img)
            else:
                attack_points = []

            center_hs = []
            center_ws = []


            attack_parallel_num, parallel_init_flag = self.attack_parallel_selection(self.attack_parallel, self.n_queries, 10000)
            # if attack_parallel_num == 2:
            #     print('test')
            if parallel_init_flag or not self.mga_flag:
                if self.mga_iters != 0:
                    self.child_windows = np.zeros([int(attack_parallel_num)*self.divide_conquer_num*self.divide_conquer_num, 2])
                else:
                    self.child_windows = None
                for _ in range(0, int(attack_parallel_num)):    
                # for _ in range(0, int(self.attack_parallel)):    
                    if len(attack_points) > 10000:                        
                        center_h, center_w = attack_points[np.random.randint(0, len(attack_points))]
                    else:
                        center_h = np.random.randint(0, h - s)
                        center_w = np.random.randint(0, w - s)

                    center_hs.append(center_h)
                    center_ws.append(center_w)
            
            if parallel_init_flag and int(attack_parallel_num) != 1:
                self.mga_flag = False

            if parallel_init_flag or not self.mga_flag:
                self.mga_center_hs = np.array(center_hs).copy()
                self.mga_center_ws = np.array(center_ws).copy()
            else:
                center_hs = self.mga_center_hs.copy()
                center_ws = self.mga_center_ws.copy()

            count_index = 0
            if parallel_init_flag or not self.mga_flag:
                self.children = np.ones([int(attack_parallel_num)*self.divide_conquer_num*self.divide_conquer_num, self.pop_size, self.pop.shape[1], child_step, child_step])
                for count, (center_h, center_w) in enumerate(zip(center_hs, center_ws)):
                    if center_h > h-s or center_w > w -s:
                        continue
                    x_window = self.x[:, :, center_h:center_h+s, center_w:center_w+s]
                    x_best_window = xs[:, :, center_h:center_h+s, center_w:center_w+s]

                    # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                    while np.sum(np.abs(np.clip(x_window + deltas[:, :, center_h:center_h+s, center_w:center_w+s], self.lb, self.ub) - x_best_window) < 10**-7) == self.pop_size*c*s*s:
                        if bool(self.divide_conquer_restart):
                            for index in range(self.pop_size):
                                deltas[index, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-self.epsilon, self.epsilon], size=[c, 1, 1])
                    for u in range(0, self.divide_conquer_num):
                        u = u * child_step
                        for v in range(0, self.divide_conquer_num):
                            v = v * child_step
                            if self.child_windows is not None:
                                self.child_windows[count_index, ] = [center_h+u, center_w+v]
                                self.children[count_index:count_index+1, :, :, 0:child_step, 0:child_step] = deltas[np.newaxis, : , :, center_h+u:center_h+u+child_step, center_w+v:center_w+v+child_step]
                            count_index += 1
                self.mga_counts = 0
                if self.child_windows is not None:
                    self.pop_children_fitness = np.ones([self.child_windows.shape[0],self.pop_size]) * 22222.0
                    self.mga_flag = True
                self.is_change = np.ones(self.pop_size)
            else:
                if int(self.pop_size) == 1:
                    self.idx = np.random.choice(np.arange(self.pop_size), size=2, replace=True)
                else:
                    self.idx = np.random.choice(np.arange(self.pop_size), size=2, replace=False)

                lw = deltas[self.idx].copy()  # short for losser winner
                mga_now_queries = self.n_queries.copy()
                
                self.best_children_fitness = self.new_loss_children.copy()

                if np.min(self.best_children_fitness) == 22222.0:
                    self.mga_flag = False

                if self.mga_flag:
                    # print('------------detector_inference-------------')
                    # start = time.time()
                    pop_child_fitness = self.get_child_fitness(lw, xs, img_metas, clean_info, loss_fct, child_step, self.child_windows)
                    # print(time.time() - start)
                    self.mga_counts = self.mga_counts +  self.n_queries - mga_now_queries
                    
                    if self.mga_counts >= self.mga_iters:
                        self.mga_flag = False

                    if self.mga_flag:

                        pop_fitness = pop_child_fitness[0]
                        child_fitness = pop_child_fitness[1]

                        if self.targeted:
                            fidx = np.argsort(-pop_fitness)
                        else:
                            fidx = np.argsort(pop_fitness)

                        lw = lw[fidx]
                        child_fitness = child_fitness[fidx]
                        
                        self.is_change_flag = np.zeros(len(fidx))
                        
                        lw = self.cross_over(lw, self.child_windows, child_step, child_fitness)
                        lw = self.mutate(lw, self.child_windows, child_step, child_fitness)

                        lw = lw[fidx]
                        child_fitness = child_fitness[fidx]
                        deltas[self.idx] = lw.copy()

                        self.is_change_flag = self.is_change_flag[fidx]

                        for flag in self.is_change_flag:
                            self.is_change[self.idx] = 1 * int(flag)

            x_new = np.clip(self.x + deltas, self.lb, self.ub).transpose(0,2,3,1)

            if self.child_windows is not None:
                # print('--------------detector_inference---------')
                # start = time.time()
                new_loss, self.new_loss_children = loss_fct(self, x_new[np.argmax(self.pop_fitness)][np.newaxis, :, :, :], img_metas, clean_info, child_step, self.child_windows, self.fitness_style)
                # print(time.time()-start)
                self.pop_children_fitness[:, np.argmax(self.pop_fitness)] = self.new_loss_children.copy()
                self.is_change[np.argmax(self.pop_fitness)] = 0.0
                stop_condition = False
                if 'child_fitness' in locals().keys():
                    window_num = (self.new_loss_children != 22222.0).sum(0)
                    stop_condition = (self.new_loss_children - child_fitness[np.argmax(self.pop_fitness)]).sum(0) < 0.01 * window_num
                if (np.min(self.new_loss_children, axis=0) < 22222.0) and (self.mga_counts < self.mga_iters):
                    self.mga_flag = True
                elif (np.min(self.new_loss_children, axis=0) == 22222.0):
                    self.mga_flag = False
                if stop_condition:
                    self.mga_flag = False
            else:
                # print('------------detector_inference-------------')
                # start = time.time()
                new_loss = loss_fct(self, x_new, img_metas, clean_info)
                # print(time.time()-start)
            # temp = np.argmax(self.pop_fitness)

            self.n_queries += np.ones(1)
            idx_improved = new_loss > self.best_fitness
            self.best_fitness = idx_improved * new_loss + ~idx_improved * self.best_fitness
            self.pop_fitness[np.argmax(self.pop_fitness)] =  self.best_fitness.copy()
                
            if self.visual_iters_path is not None:
                self.iters_list.append(self.iters_list[-1]+1)
                self.losses_list[0].append(float(new_loss))
                self.losses_list[1].append(float(self.best_fitness))

            xs = xs.transpose(0,2,3,1)
            idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x_new.shape[:-1])])
            x_new[np.argmax(self.pop_fitness)] = idx_improved * x_new[np.argmax(self.pop_fitness)] + ~idx_improved * xs[np.argmax(self.pop_fitness)]
            
            return t(x_new), self.n_queries-now_queries

