"""
Implements Square attacks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch import Tensor as t

from blackbox_attack.black_box_attack import BlackBoxAttack
from blackbox_attack.utils.compute_fcts import lp_step

class SquareAttack(BlackBoxAttack):
    """
    Square Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, p_init, lb, ub, name, attack_model, attack_mode, attack_logistics, loss, targeted, ori_img, model_name):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_crit_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         name="square",
                         attack_model=attack_model,
                         attack_logistics=attack_logistics,
                         attack_mode=attack_mode,
                         loss=loss,
                         targeted=targeted,
                         ori_img=ori_img,
                         model_name=model_name)
        self.best_loss = None
        self.i = 0
        self.p_init = p_init

    def p_selection(self, p_init, it, n_iters):
        """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
        it = int(it / n_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p

    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = np.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
                max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1

        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        return delta


    def meta_pseudo_gaussian_pert(self, s):
        delta = np.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
            if np.random.rand(1) > 0.5: delta = np.transpose(delta)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

        return delta

    def _suggest(self, xs_t, loss_fct, img_metas, clean_info):
        xs = xs_t.cpu().numpy().transpose(0,3,1,2)
        c, h, w = xs.shape[1:]
        n_features = c*h*w
        n_queries = np.zeros(xs.shape[0])

        if self.p == 'inf':
            if self.is_new_batch:
                self.x = xs.copy()
                init_delta = np.random.choice([-self.epsilon, self.epsilon], size=[xs.shape[0], c, 1, w])
                xs = np.clip(xs + init_delta, self.lb, self.ub)
                self.best_loss = loss_fct(self, xs.transpose(0,2,3,1), img_metas, clean_info)
                n_queries += np.ones(xs.shape[0])
                self.i = 0

            deltas = xs - self.x
            p = self.p_selection(self.p_init, self.i, 10000)
            for i_img in range(xs.shape[0]):
                s = int(round(np.sqrt(p * n_features / c)))
                s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)

                x_window = self.x[i_img, :, center_h:center_h+s, center_w:center_w+s]
                x_best_window = xs[i_img, :, center_h:center_h+s, center_w:center_w+s]
                # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
                while np.sum(np.abs(np.clip(x_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], self.lb, self.ub) - x_best_window) < 10**-7) == c*s*s:
                    deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-self.epsilon, self.epsilon], size=[c, 1, 1])

            x_new = np.clip(self.x + deltas, self.lb, self.ub).transpose(0,2,3,1)

        elif self.p == '2':
            if self.is_new_batch:
                self.x = xs.copy()
                delta_init = np.zeros(xs.shape)
                s = h // 5
                sp_init = (h - s * 5) // 2
                center_h = sp_init + 0
                for _ in range(h // s):
                    center_w = sp_init + 0
                    for _ in range(w // s):
                        delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                            [1, 1, s, s]) * np.random.choice([-1, 1], size=[xs.shape[0], c, 1, 1])
                        center_w += s
                    center_h += s
                xs = np.clip(xs + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * (self.epsilon), self.lb, self.ub) 
                self.best_loss = loss_fct(self, xs.transpose(0,2,3,1), img_metas, clean_info)
                n_queries += np.ones(xs.shape[0])
                self.i = 0

            deltas = xs - self.x
            p = self.p_selection(self.p_init, self.i, 10000)
            s = max(int(round(np.sqrt(p * n_features / c))), 3)
            if s % 2 == 0:
                s += 1

            s2 = s + 0
            ### window_1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            new_deltas_mask = np.zeros(xs.shape)
            new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

            ### window_2
            center_h_2 = np.random.randint(0, h - s2)
            center_w_2 = np.random.randint(0, w - s2)
            new_deltas_mask_2 = np.zeros(xs.shape)
            new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0

            ### compute total norm available
            curr_norms_window = np.sqrt(
                np.sum(((xs - self.x) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
            curr_norms_image = np.sqrt(np.sum((xs - self.x) ** 2, axis=(1, 2, 3), keepdims=True))
            mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
            norms_windows = np.sqrt(np.sum((deltas * mask_2) ** 2, axis=(2, 3), keepdims=True))

            ### create the updates
            new_deltas = np.ones([self.x.shape[0], c, s, s])
            new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
            new_deltas *= np.random.choice([-1, 1], size=[self.x.shape[0], c, 1, 1])
            old_deltas = deltas[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
            new_deltas += old_deltas
            new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum((self.epsilon) ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
            deltas[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
            deltas[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

            x_new = self.x + deltas / np.sqrt(np.sum(deltas ** 2, axis=(1, 2, 3), keepdims=True)) * (self.epsilon)
            x_new = np.clip(x_new, self.lb, self.ub).transpose(0,2,3,1)


        new_loss = loss_fct(self, x_new, img_metas, clean_info)
        n_queries += np.ones(xs.shape[0])
        idx_improved = new_loss > self.best_loss
        self.best_loss = idx_improved * new_loss + ~idx_improved * self.best_loss
        xs = xs.transpose(0,2,3,1)
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x_new.shape[:-1])])
        x_new = idx_improved * x_new + ~idx_improved * xs
        self.i += 1
        
        return t(x_new), n_queries

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "max_crit_queries": "inf" if np.isinf(self.max_crit_queries) else self.max_crit_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "attack_name": self.__class__.__name__
        }
