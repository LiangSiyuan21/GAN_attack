import torch
import torch.nn as nn
import torch.nn.functional as F

from attack import Attack


class DI2FGSM(Attack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 8/255)
        alpha (float): step size. (DEFAULT: 2/255)
        decay (float): momentum factor. (DEFAULT: 0.0)
        steps (int): number of iterations. (DEFAULT: 20)
        resize_rate (float): resize factor used in input diversity. (DEFAULT: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (DEFAULT: 0.5)
        random_start (bool): using random initialization of delta. (DEFAULT: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        # DI2FGSM
        >>> attack = torchattacks.DI2FGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        >>> adv_images = attack(images, labels)
        # M-DI2FGSM
        >>> attack = torchattacks.DI2FGSM(model, eps=8/255, alpha=2/255, steps=20, decay=1.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False, ub=1, lb=0):
        super(DI2FGSM, self).__init__("DI2FGSM", model)
        self.eps = eps *(ub-lb)
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.ub = ub
        self.lb = lb

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
        images = data['img'][0].clone().detach().to(self.device)
        
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        new_data = {}
        new_data['img_metas'] = data['img_metas'][0]
        for i in range(self.steps):
            adv_images.requires_grad = True
            data['img'][0] = adv_images
            
            new_data['img'] = adv_images

            losses = self.model(return_loss=True, **new_data, gt_bboxes=gts[0].unsqueeze(0), gt_labels=gts[1].unsqueeze(0))
            
            self.model.zero_grad()
            losses['loss_cls'] = losses['loss_cls'] * (-1.0)
            losses['loss_cls'].backward()
            grad = adv_images.grad.data
            
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=self.lb, max=self.ub).detach()

        return adv_images